"""
跨进程共享版本的全局通信时延收集器

该模块提供一个全局单例收集器，用于:
1. 在 vLLM 的 all2all.py 中记录通信时延（Worker 子进程）
2. 在训练模块中获取这些时延数据用于计算奖励（主进程）

【重要】跨进程数据共享机制：
- 使用 multiprocessing.shared_memory.SharedMemory 创建命名共享内存
- 通过固定的名字 "comm_delay_shm" 在独立进程间共享同一块内存
- Worker 子进程中调用 record_dispatch/combine 会写入共享内存
- 主进程中调用 get_total_delay 会从同一块共享内存中读取
- 适用于 torchrun/spawn 等独立进程启动方式

使用方式:
---------
在主进程中（启动时）:
    from moe_route_optimizer.hooks.comm_delay_collector import init_shared_memory
    init_shared_memory()  # 必须在启动 vLLM 之前调用！

在 vLLM all2all.py 中（Worker 子进程）:
    from moe_route_optimizer.hooks.comm_delay_collector import get_collector
    collector = get_collector()
    collector.record_dispatch(elapsed_time_ms, layer_idx=0)

在训练模块中（主进程）:
    from moe_route_optimizer.hooks.comm_delay_collector import get_collector
    collector = get_collector()
    total_delay = collector.get_total_delay()  # 可以读取到子进程写入的数据
    collector.reset()  # 每次推理后重置

清理:
    from moe_route_optimizer.hooks.comm_delay_collector import cleanup_shared_memory
    cleanup_shared_memory()  # 程序退出时调用
"""

import threading
import struct
import os
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
import fcntl


@dataclass
class CommDelayRecord:
    """单次通信记录"""
    timestamp: float          # 记录时间戳
    delay_ms: float           # 时延（毫秒）
    operation: str            # 操作类型: "dispatch" 或 "combine"
    layer_idx: int = 0        # MoE层索引
    call_count: int = 0       # 调用计数
    

# ============================================================
# 命名共享内存（支持独立进程间共享）
# ============================================================
# 共享内存布局（32字节）:
#   - 8 bytes: total_delay_ms (double)
#   - 4 bytes: dispatch_count (int32)
#   - 4 bytes: combine_count (int32)
#   - 16 bytes: 保留/对齐

SHM_NAME = "moe_comm_delay_collector"  # 共享内存名称
SHM_SIZE = 32  # 共享内存大小（字节）
LOCK_FILE = "/tmp/moe_comm_delay_collector.lock"  # 文件锁路径

# 全局变量
_shm: Optional[shared_memory.SharedMemory] = None
_is_creator = False  # 当前进程是否是共享内存的创建者


def _acquire_file_lock():
    """获取文件锁（跨进程互斥）"""
    # 确保锁文件存在
    if not os.path.exists(LOCK_FILE):
        try:
            open(LOCK_FILE, 'w').close()
        except:
            pass
    try:
        lock_fd = open(LOCK_FILE, 'r+')
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        return lock_fd
    except:
        return None


def _release_file_lock(lock_fd):
    """释放文件锁"""
    if lock_fd:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()
        except:
            pass


def init_shared_memory():
    """
    初始化共享内存（主进程调用）
    
    必须在启动 vLLM worker 进程之前调用！
    创建一块命名共享内存，worker 进程可以通过名字访问。
    """
    global _shm, _is_creator
    
    if _shm is not None:
        return  # 已初始化
    
    try:
        # 尝试创建新的共享内存
        _shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
        _is_creator = True
        # 初始化为0
        _shm.buf[:SHM_SIZE] = bytes(SHM_SIZE)
        print(f"[CommCollector] Created shared memory: {SHM_NAME}")
    except FileExistsError:
        # 共享内存已存在，连接到它
        _shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
        _is_creator = False
        print(f"[CommCollector] Connected to existing shared memory: {SHM_NAME}")


def _ensure_shm_connected():
    """确保已连接到共享内存（worker 进程自动调用）"""
    global _shm, _is_creator
    
    if _shm is not None:
        return True
    
    try:
        # 尝试连接到已存在的共享内存
        _shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
        _is_creator = False
        return True
    except FileNotFoundError:
        # 共享内存不存在，尝试创建（兜底）
        try:
            _shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
            _is_creator = True
            _shm.buf[:SHM_SIZE] = bytes(SHM_SIZE)
            return True
        except:
            return False
    except Exception as e:
        print(f"[CommCollector] Failed to connect to shared memory: {e}")
        return False


def cleanup_shared_memory():
    """
    清理共享内存（主进程退出时调用）
    """
    global _shm, _is_creator
    
    if _shm is not None:
        try:
            _shm.close()
            if _is_creator:
                _shm.unlink()  # 只有创建者才能 unlink
                print(f"[CommCollector] Cleaned up shared memory: {SHM_NAME}")
        except:
            pass
        _shm = None
    
    # 清理锁文件
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except:
        pass


class CommDelayCollector:
    """
    通信时延收集器（跨进程版本 - 使用命名共享内存）
    
    使用 multiprocessing.shared_memory.SharedMemory 支持 vLLM 多进程场景。
    通过固定的名字在完全独立的进程间共享同一块内存。
    适用于 torchrun/spawn 等独立进程启动方式。
    
    共享内存布局 (32 bytes):
    - offset 0-7:   total_delay_ms (double, 8 bytes)
    - offset 8-11:  dispatch_count (int32, 4 bytes)
    - offset 12-15: combine_count (int32, 4 bytes)
    - offset 16-31: reserved
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._data_lock = threading.Lock()  # 线程锁（进程内）
        
        # 存储当前推理周期的记录（进程本地，详细记录不跨进程共享）
        self._current_records: List[CommDelayRecord] = []
        
        # 按层和操作类型分类的累计时延（进程本地）
        self._dispatch_delays: Dict[int, List[float]] = defaultdict(list)
        self._combine_delays: Dict[int, List[float]] = defaultdict(list)
        
        # 本地计数器（用于详细记录）
        self._local_dispatch_count = 0
        self._local_combine_count = 0
        
        # 是否启用收集
        self._enabled = True
        
        # 回调函数（可选）
        self._callbacks: List[callable] = []
        
        # 调试模式
        self._debug = False
    
    def enable(self):
        """启用收集"""
        self._enabled = True
    
    def disable(self):
        """禁用收集"""
        self._enabled = False
    
    def set_debug(self, debug: bool):
        """设置调试模式"""
        self._debug = debug
    
    def _read_shared_data(self) -> Tuple[float, int, int]:
        """从共享内存读取数据"""
        if not _ensure_shm_connected():
            return (0.0, 0, 0)
        
        lock_fd = _acquire_file_lock()
        try:
            # 读取: double (8 bytes) + int (4 bytes) + int (4 bytes)
            data = bytes(_shm.buf[:16])
            total_delay, dispatch_count, combine_count = struct.unpack('dii', data)
            return (total_delay, dispatch_count, combine_count)
        finally:
            _release_file_lock(lock_fd)
    
    def _write_shared_data(self, total_delay: float, dispatch_count: int, combine_count: int):
        """写入数据到共享内存"""
        if not _ensure_shm_connected():
            return
        
        lock_fd = _acquire_file_lock()
        try:
            data = struct.pack('dii', total_delay, dispatch_count, combine_count)
            _shm.buf[:16] = data
        finally:
            _release_file_lock(lock_fd)
    
    def _atomic_add(self, delay_ms: float, is_dispatch: bool) -> Tuple[float, int]:
        """原子地增加时延和计数器"""
        if not _ensure_shm_connected():
            return (0.0, 0)
        
        lock_fd = _acquire_file_lock()
        try:
            # 读取当前值
            data = bytes(_shm.buf[:16])
            total_delay, dispatch_count, combine_count = struct.unpack('dii', data)
            
            # 更新
            total_delay += delay_ms
            if is_dispatch:
                dispatch_count += 1
                current_count = dispatch_count
            else:
                combine_count += 1
                current_count = combine_count
            
            # 写回
            new_data = struct.pack('dii', total_delay, dispatch_count, combine_count)
            _shm.buf[:16] = new_data
            
            return (total_delay, current_count)
        finally:
            _release_file_lock(lock_fd)
    
    def record_dispatch(self, delay_ms: float, layer_idx: int = 0):
        """
        记录一次 dispatch (all-gather) 通信时延
        
        Args:
            delay_ms: 时延（毫秒）
            layer_idx: MoE层索引
        """
        if not self._enabled:
            return
        
        # 原子地更新共享内存
        shared_total, current_count = self._atomic_add(delay_ms, is_dispatch=True)
        
        # 本地计数器
        self._local_dispatch_count += 1
        
        # 本地记录（详细信息）
        record = CommDelayRecord(
            timestamp=time.time(),
            delay_ms=delay_ms,
            operation="dispatch",
            layer_idx=layer_idx,
            call_count=current_count
        )
        self._current_records.append(record)
        self._dispatch_delays[layer_idx].append(delay_ms)
        
        if self._debug:
            print(f"[CommCollector] Recorded dispatch: {delay_ms:.3f}ms (layer={layer_idx}, count={current_count}, shared_total={shared_total:.3f}ms)")
        
        # 触发回调
        for callback in self._callbacks:
            try:
                callback("dispatch", delay_ms, layer_idx)
            except Exception:
                pass
    
    def record_combine(self, delay_ms: float, layer_idx: int = 0):
        """
        记录一次 combine (reduce-scatter) 通信时延
        
        Args:
            delay_ms: 时延（毫秒）
            layer_idx: MoE层索引
        """
        if not self._enabled:
            return
        
        # 原子地更新共享内存
        shared_total, current_count = self._atomic_add(delay_ms, is_dispatch=False)
        
        # 本地计数器
        self._local_combine_count += 1
        
        # 本地记录（详细信息）
        record = CommDelayRecord(
            timestamp=time.time(),
            delay_ms=delay_ms,
            operation="combine",
            layer_idx=layer_idx,
            call_count=current_count
        )
        self._current_records.append(record)
        self._combine_delays[layer_idx].append(delay_ms)
        
        if self._debug:
            print(f"[CommCollector] Recorded combine: {delay_ms:.3f}ms (layer={layer_idx}, count={current_count}, shared_total={shared_total:.3f}ms)")
        
        # 触发回调
        for callback in self._callbacks:
            try:
                callback("combine", delay_ms, layer_idx)
            except Exception:
                pass
    
    def get_total_delay(self) -> float:
        """
        获取当前推理周期的总通信时延（毫秒）
        从命名共享内存中读取，支持跨进程访问。
        
        Returns:
            总时延（毫秒）
        """
        total_delay, _, _ = self._read_shared_data()
        return total_delay
    
    def get_total_delay_seconds(self) -> float:
        """
        获取当前推理周期的总通信时延（秒）
        
        Returns:
            总时延（秒）
        """
        return self.get_total_delay() / 1000.0
    
    def get_dispatch_count(self) -> int:
        """获取共享的 dispatch 计数"""
        _, dispatch_count, _ = self._read_shared_data()
        return dispatch_count
    
    def get_combine_count(self) -> int:
        """获取共享的 combine 计数"""
        _, _, combine_count = self._read_shared_data()
        return combine_count
    
    def get_dispatch_delay(self) -> float:
        """获取当前周期所有 dispatch 的总时延（毫秒）- 本地进程数据"""
        with self._data_lock:
            total = 0.0
            for delays in self._dispatch_delays.values():
                total += sum(delays)
            return total
    
    def get_combine_delay(self) -> float:
        """获取当前周期所有 combine 的总时延（毫秒）"""
        with self._data_lock:
            total = 0.0
            for delays in self._combine_delays.values():
                total += sum(delays)
            return total
    
    def get_delay_per_layer(self) -> Dict[int, float]:
        """
        获取每层的总通信时延
        
        Returns:
            {layer_idx: total_delay_ms, ...}
        """
        with self._data_lock:
            result = {}
            all_layers = set(self._dispatch_delays.keys()) | set(self._combine_delays.keys())
            for layer_idx in all_layers:
                dispatch_total = sum(self._dispatch_delays.get(layer_idx, []))
                combine_total = sum(self._combine_delays.get(layer_idx, []))
                result[layer_idx] = dispatch_total + combine_total
            return result
    
    def get_records(self) -> List[CommDelayRecord]:
        """获取当前周期的所有记录"""
        with self._data_lock:
            return list(self._current_records)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取当前周期的统计信息
        包含命名共享内存中的全局数据和本地进程的详细数据
        
        Returns:
            包含各项统计的字典
        """
        # 获取共享内存中的全局数据
        shared_total, shared_dispatch, shared_combine = self._read_shared_data()
        
        # 获取本地详细数据
        with self._data_lock:
            dispatch_delays_all = []
            combine_delays_all = []
            for delays in self._dispatch_delays.values():
                dispatch_delays_all.extend(delays)
            for delays in self._combine_delays.values():
                combine_delays_all.extend(delays)
            
            stats = {
                # 共享内存中的全局数据（跨进程累加）
                'total_delay_ms': shared_total,
                'total_delay_s': shared_total / 1000.0,
                'dispatch_count': shared_dispatch,
                'combine_count': shared_combine,
                'total_count': shared_dispatch + shared_combine,
                # 本地进程的详细数据
                'local_dispatch_total_ms': sum(dispatch_delays_all),
                'local_combine_total_ms': sum(combine_delays_all),
                'local_dispatch_count': self._local_dispatch_count,
                'local_combine_count': self._local_combine_count,
            }
            
            if dispatch_delays_all:
                stats['dispatch_avg_ms'] = sum(dispatch_delays_all) / len(dispatch_delays_all)
                stats['dispatch_max_ms'] = max(dispatch_delays_all)
                stats['dispatch_min_ms'] = min(dispatch_delays_all)
            
            if combine_delays_all:
                stats['combine_avg_ms'] = sum(combine_delays_all) / len(combine_delays_all)
                stats['combine_max_ms'] = max(combine_delays_all)
                stats['combine_min_ms'] = min(combine_delays_all)
            
            return stats
    
    def reset(self):
        """
        重置当前周期的数据
        在每次推理完成后调用，准备收集下一次推理的数据
        
        注意：会重置命名共享内存中的全局数据和本地数据
        """
        # 重置共享内存中的全局数据
        self._write_shared_data(0.0, 0, 0)
        
        # 重置本地数据
        with self._data_lock:
            self._current_records.clear()
            self._dispatch_delays.clear()
            self._combine_delays.clear()
            self._local_dispatch_count = 0
            self._local_combine_count = 0
            
            if self._debug:
                print("[CommCollector] Reset (shared + local)")
    
    def add_callback(self, callback: callable):
        """
        添加回调函数，在每次记录时触发
        
        Args:
            callback: 回调函数，签名为 callback(operation: str, delay_ms: float, layer_idx: int)
        """
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: callable):
        """移除回调函数"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def clear_callbacks(self):
        """清除所有回调函数"""
        self._callbacks.clear()


# 全局实例获取函数
def get_collector() -> CommDelayCollector:
    """
    获取全局通信时延收集器实例
    
    Returns:
        CommDelayCollector 单例
    """
    return CommDelayCollector()


def reset_collector():
    """重置收集器数据（便捷函数）"""
    get_collector().reset()


def get_total_comm_delay() -> float:
    """获取总通信时延（秒）（便捷函数）"""
    return get_collector().get_total_delay_seconds()


def get_comm_statistics() -> Dict[str, float]:
    """获取通信统计信息（便捷函数）"""
    return get_collector().get_statistics()
