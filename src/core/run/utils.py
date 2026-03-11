"""`src.core.run` 子包通用辅助工具。"""
from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path


def _cleanup_data_files(output_dir: str, log_dir: Path):
    """清理数据文件，只保留日志"""
    output_path = Path(output_dir)
    
    # 复制日志文件到log_dir
    for log_file in output_path.glob("**/*.txt"):
        dest = log_dir / log_file.name
        shutil.copy2(log_file, dest)
        print(f"  📄 保存日志: {log_file.name}")
    
    for log_file in output_path.glob("**/*.log"):
        dest = log_dir / log_file.name
        shutil.copy2(log_file, dest)
        print(f"  📄 保存日志: {log_file.name}")
    
    # 强制垃圾回收，释放可能被 pandas 持有的文件句柄
    gc.collect()
    
    # 删除整个临时输出目录（带重试机制）
    temp_dir = output_path.parent
    max_retries = 3
    for attempt in range(max_retries):
        try:
            shutil.rmtree(temp_dir)
            print(f"  [DEL] 已清理临时数据目录")
            break
        except PermissionError as e:
            if attempt < max_retries - 1:
                # 等待一小段时间让文件句柄释放
                time.sleep(0.5)
                gc.collect()
            else:
                # 最后一次尝试失败，尝试逐个删除文件
                print(
                    f"  [WARN] 临时目录清理延迟"
                    f"（文件可能被占用）: {temp_dir}"
                )
                try:
                    # 尝试删除可以删除的文件
                    for file in temp_dir.rglob("*"):
                        if file.is_file():
                            try:
                                file.unlink()
                            except:
                                pass
                except:
                    pass
        except Exception as e:
            print(f"  [WARN] 清理临时目录失败: {e}")
            break
