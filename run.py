# 运行完整集成仿真
from main_integration import run_integrated_simulation

result = run_integrated_simulation(
    config_path="./config/supply_chain_config.xlsx",
    start_date="2024-01-01",
    end_date="2024-01-07", 
    output_base_dir="./integrated_output"
)