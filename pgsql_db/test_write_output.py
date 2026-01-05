"""测试模块输出写入数据库"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pgsql_db.db_connection import DatabaseConnection
from pgsql_db.module_data_writer import ModuleDataWriter

def main():
    db = DatabaseConnection()
    print('测试数据库连接...')
    result = db.test_connection()
    print(f'连接状态: {"成功" if result["success"] else "失败"}')
    
    if result['success']:
        print('\n写入模块输出数据...')
        writer = ModuleDataWriter(db)
        writer.write_all_modules('integrated_output')
        
        # 写入orchestrator数据
        print('\n写入Orchestrator数据...')
        writer.write_orchestrator_data('integrated_output/orchestrator')
        
        writer.print_summary()
    
    db.close()
    print('\n✅测试完成')

if __name__ == "__main__":
    main()
