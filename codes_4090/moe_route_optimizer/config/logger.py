"""
日志管理模块
统一管理训练和评估日志的输出
支持分布式环境下只有rank 0输出日志
"""

import logging
import sys
import os
from typing import Optional
from datetime import datetime


def get_rank() -> int:
    """
    获取当前进程的rank
    支持torch.distributed和环境变量两种方式
    
    Returns:
        当前进程的rank，非分布式环境返回0
    """
    # 尝试从torch.distributed获取
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except:
        pass
    
    # 尝试从环境变量获取（torchrun设置）
    rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))
    return int(rank)


def is_main_process() -> bool:
    """
    判断当前进程是否是主进程（rank 0）
    
    Returns:
        是否是主进程
    """
    return get_rank() == 0


class RankFilter(logging.Filter):
    """
    日志过滤器，只允许rank 0的进程输出日志
    """
    def __init__(self, rank: int = 0):
        super().__init__()
        self.rank = rank
    
    def filter(self, record):
        return get_rank() == self.rank


class LoggerManager:
    """日志管理器，支持分布式环境"""
    
    _instance = None
    _loggers = {}
    _rank = 0
    _is_main = True
    _session_log_dir = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def setup(cls, log_dir: str, train_log_file: str = "training.log", 
              eval_log_file: str = "evaluation.log", debug: bool = False,
              rank: Optional[int] = None):
        """
        初始化日志系统
        
        Args:
            log_dir: 日志目录
            train_log_file: 训练日志文件名
            eval_log_file: 评估日志文件名
            debug: 是否开启debug模式
            rank: 当前进程rank（None则自动检测）
        """
        # 获取rank
        cls._rank = rank if rank is not None else get_rank()
        cls._is_main = (cls._rank == 0)

        # 每次启动创建独立的时间目录，避免不同运行覆盖彼此日志
        if cls._is_main and cls._session_log_dir is None:
            session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cls._session_log_dir = os.path.join(log_dir, session_name)
        elif cls._session_log_dir is None:
            cls._session_log_dir = log_dir
        
        # 只有rank 0创建日志目录和文件
        if cls._is_main:
            os.makedirs(cls._session_log_dir, exist_ok=True)
        
        # 日志格式（包含rank信息）
        log_format = logging.Formatter(
            f'[%(asctime)s] [Rank {cls._rank}] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        level = logging.DEBUG if debug else logging.INFO
        
        # 创建rank过滤器（只允许rank 0输出）
        rank_filter = RankFilter(rank=0)
        
        # 创建训练日志器
        train_logger = logging.getLogger('train')
        train_logger.setLevel(level)
        train_logger.handlers.clear()
        train_logger.addFilter(rank_filter)
        
        if cls._is_main:
            # 文件handler（只有rank 0写文件）
            train_file_handler = logging.FileHandler(
                os.path.join(cls._session_log_dir, train_log_file), 
                encoding='utf-8'
            )
            train_file_handler.setFormatter(log_format)
            train_logger.addHandler(train_file_handler)
            
            # 控制台handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(log_format)
            train_logger.addHandler(console_handler)
        
        cls._loggers['train'] = train_logger
        
        # 创建评估日志器
        eval_logger = logging.getLogger('eval')
        eval_logger.setLevel(level)
        eval_logger.handlers.clear()
        eval_logger.addFilter(rank_filter)
        
        if cls._is_main:
            eval_file_handler = logging.FileHandler(
                os.path.join(cls._session_log_dir, eval_log_file),
                encoding='utf-8'
            )
            eval_file_handler.setFormatter(log_format)
            eval_logger.addHandler(eval_file_handler)
            eval_logger.addHandler(console_handler)
        
        cls._loggers['eval'] = eval_logger
        
        # 创建通用日志器
        general_logger = logging.getLogger('general')
        general_logger.setLevel(level)
        general_logger.handlers.clear()
        general_logger.addFilter(rank_filter)
        
        if cls._is_main:
            general_file_handler = logging.FileHandler(
                os.path.join(cls._session_log_dir, "general.log"),
                encoding='utf-8'
            )
            general_file_handler.setFormatter(log_format)
            general_logger.addHandler(general_file_handler)
            general_logger.addHandler(console_handler)
        
        cls._loggers['general'] = general_logger
        
        if cls._is_main:
            general_logger.info(
                f"Logger initialized. Log directory: {cls._session_log_dir}, Rank: {cls._rank}"
            )
    
    @classmethod
    def get_rank(cls) -> int:
        """获取当前rank"""
        return cls._rank
    
    @classmethod
    def is_main_process(cls) -> bool:
        """是否是主进程"""
        return cls._is_main
    
    @classmethod
    def get_logger(cls, name: str = 'general') -> logging.Logger:
        """
        获取日志器
        
        Args:
            name: 日志器名称 ('train', 'eval', 'general')
        
        Returns:
            Logger对象
        """
        if name not in cls._loggers:
            # 如果未初始化，返回默认logger
            logger = logging.getLogger(name)
            if not logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(logging.Formatter(
                    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
                ))
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
            return logger
        return cls._loggers[name]


def get_train_logger() -> logging.Logger:
    """获取训练日志器的便捷方法"""
    return LoggerManager.get_logger('train')


def get_eval_logger() -> logging.Logger:
    """获取评估日志器的便捷方法"""
    return LoggerManager.get_logger('eval')


def get_logger() -> logging.Logger:
    """获取通用日志器的便捷方法"""
    return LoggerManager.get_logger('general')


class TrainingMetricsLogger:
    """训练指标记录器"""
    
    def __init__(self, log_interval: int = 5):
        self.logger = get_train_logger()
        self.log_interval = log_interval
        self.step = 0
        self.episode = 0
        
        # 累积指标
        self.accumulated_rewards = []
        self.accumulated_latency_reductions = []
        self.accumulated_accuracies = []
        self.accumulated_actor_losses = []
        self.accumulated_critic_losses = []
    
    def log_step(self, reward: float, latency_reduction: float, 
                 accuracy: float, actor_loss: float = 0.0, 
                 critic_loss: float = 0.0):
        """记录单步训练指标"""
        self.step += 1
        self.accumulated_rewards.append(reward)
        self.accumulated_latency_reductions.append(latency_reduction)
        self.accumulated_accuracies.append(accuracy)
        self.accumulated_actor_losses.append(actor_loss)
        self.accumulated_critic_losses.append(critic_loss)
        
        if self.step % self.log_interval == 0:
            avg_reward = sum(self.accumulated_rewards[-self.log_interval:]) / self.log_interval
            avg_latency = sum(self.accumulated_latency_reductions[-self.log_interval:]) / self.log_interval
            avg_acc = sum(self.accumulated_accuracies[-self.log_interval:]) / self.log_interval
            avg_actor_loss = sum(self.accumulated_actor_losses[-self.log_interval:]) / self.log_interval
            avg_critic_loss = sum(self.accumulated_critic_losses[-self.log_interval:]) / self.log_interval
            
            self.logger.info(
                f"Step {self.step} | "
                f"Reward: {avg_reward:.4f} | "
                f"Latency↓: {avg_latency*100:.2f}% | "
                f"Acc: {avg_acc*100:.2f}% | "
                f"Actor Loss: {avg_actor_loss:.4f} | "
                f"Critic Loss: {avg_critic_loss:.4f}"
            )
    
    def log_episode(self, episode: int, total_reward: float, 
                    avg_latency_reduction: float, avg_accuracy: float,
                    is_success: bool = False, extra_info: str = ""):
        """记录episode级别的指标"""
        self.episode = episode
        status = "✓" if is_success else "○"
        
        self.logger.info(
            f"Episode {episode} [{status}] | "
            f"Total Reward: {total_reward:.4f} | "
            f"Avg Latency↓: {avg_latency_reduction*100:.2f}% | "
            f"Avg Acc: {avg_accuracy*100:.2f}%"
            f"{' | ' + extra_info if extra_info else ''}"
        )
    
    def log_convergence(self, success: bool, reason: str, 
                        final_metrics: dict):
        """记录收敛结果"""
        if success:
            self.logger.info("=" * 60)
            self.logger.info("Training CONVERGED Successfully!")
            self.logger.info(f"Reason: {reason}")
            self.logger.info(f"Final Metrics: {final_metrics}")
            self.logger.info("=" * 60)
        else:
            self.logger.warning("=" * 60)
            self.logger.warning("Training STOPPED without convergence")
            self.logger.warning(f"Reason: {reason}")
            self.logger.warning(f"Final Metrics: {final_metrics}")
            self.logger.warning("=" * 60)
    
    def log_checkpoint_saved(self, path: str):
        """记录检查点保存"""
        self.logger.info(f"Checkpoint saved: {path}")


class EvaluationLogger:
    """评估日志记录器"""
    
    def __init__(self):
        self.logger = get_eval_logger()
    
    def log_baseline(self, baseline_latency: float, baseline_accuracy: float):
        """记录基线指标"""
        self.logger.info("=" * 40)
        self.logger.info("Baseline Evaluation:")
        self.logger.info(f"  Latency: {baseline_latency:.4f}s")
        self.logger.info(f"  Accuracy: {baseline_accuracy*100:.2f}%")
        self.logger.info("=" * 40)
    
    def log_evaluation(self, episode: int, latency: float, accuracy: float,
                       latency_reduction: float, accuracy_ratio: float):
        """记录评估结果"""
        self.logger.info(
            f"Evaluation @ Episode {episode} | "
            f"Latency: {latency:.4f}s (↓{latency_reduction*100:.2f}%) | "
            f"Accuracy: {accuracy*100:.2f}% ({accuracy_ratio*100:.2f}% of baseline)"
        )
    
    def log_final_evaluation(self, metrics: dict):
        """记录最终评估结果"""
        self.logger.info("=" * 40)
        self.logger.info("Final Evaluation Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 40)
