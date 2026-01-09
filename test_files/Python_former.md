# Python代码规范文档

---

## 一、代码风格与格式

### 1.1 PEP 8基础规范

- 缩进使用4个空格，禁止使用Tab
- 每行最大长度不超过79字符（文档/注释72字符）
- 运算符前后、逗号后添加空格
- 避免行尾空格，文件末尾留一个空行

### 1.2 命名约定

- 变量/函数：snake_case（lower_case_with_underscores）
- 类名：CapWords（大驼峰式）
- 常量：UPPER_CASE_WITH_UNDERSCORES
- 私有成员：前导下划线`_private`
- 避免使用单字符名称（除了迭代变量）

### 1.3 导入规范

- 按标准库、第三方库、本地库分组
- 每组内按字母顺序排列
- 避免通配符导入（from module import *）
- 相对导入明确（from . import module）

## 二、代码质量与最佳实践

### 2.1 函数设计原则

- 函数保持单一职责，控制行数（严格要求<50行）
- 参数数量适度（建议≤5个）
- 明确参数类型和返回值类型注解
- 避免函数副作用，必要时明确文档说明

### 2.2 异常处理

- 使用具体异常类型，避免裸except
- 合理使用try-except-else-finally结构
- 异常信息明确，有助于调试
- 适当使用自定义异常类

### 2.3 性能与内存

- 避免不必要的全局变量
- 使用生成器处理大数据集
- 合理使用缓存（lru_cache）
- 注意循环内的资源创建

### 2.4 装饰器使用顺序规范

- 装饰器按照从下到上（从内到外）的顺序执行

- 常用装饰器类别及推荐顺序
  
  ```
  # 1. 类装饰器（最外层）
  # 2. 属性装饰器（property, staticmethod, classmethod）
  # 3. 功能装饰器（缓存、日志、认证等）
  # 4. 验证装饰器（参数检查、类型检查等）
  # 5. 核心业务逻辑（最内层）
  
  class UserService:
      # 正确的装饰器顺序示例
      @classmethod                    # 2. 类方法装饰器
      @cache_result(ttl=3600)         # 3. 缓存装饰器
      @log_execution_time             # 3. 日志装饰器  
      @validate_arguments             # 4. 参数验证装饰器
      def get_user_profile(cls, user_id: int) -> dict:
          """5. 核心业务逻辑（最内层）"""
          # 数据库查询等核心逻辑
          return db.query(User).filter_by(id=user_id).first()
  ```

- Flask/Django 装饰器顺序规范
  
  ```
  # ✅ 正确的装饰器顺序（从外到内）：
  # 1. 路由装饰器（最外层）
  # 2. 认证/授权装饰器
  # 3. 限流装饰器
  # 4. 日志装饰器
  # 5. 验证装饰器（最内层，紧邻函数）
  
  @app.route('/api/users', methods=['POST'])
  @require_auth                    # 2. 认证（先认证再限流）
  @rate_limit(max_requests=10)    # 3. 限流
  @log_request                    # 4. 日志
  @validate_json({                # 5. 验证（最内层）
      'name': str,
      'email': str,
      'age': int
  })
  def create_user():
      """6. 核心业务逻辑"""
      data = request.get_json()
      # 创建用户逻辑
      return jsonify({"message": "用户创建成功", "data": data}), 201
  
  
  ```

- Python 内置装饰器顺序
  
  ```
  from functools import lru_cache, wraps
  from typing import Callable, List
  
  class DataProcessor:
      # ✅ 正确的顺序
      @staticmethod                    # 1. 类装饰器
      @lru_cache(maxsize=128)         # 2. 缓存装饰器
      def process_data(data: List[int]) -> List[int]:
          """处理数据 - 核心逻辑"""
          return sorted(data)
      
      # ❌ 错误的顺序
      @lru_cache(maxsize=128)
      @staticmethod  # 错误：@staticmethod 应该在 @lru_cache 上面
      def wrong_order(data: List[int]) -> List[int]:
          return sorted(data)
  
  # 属性装饰器的正确顺序
  class User:
      def __init__(self, name):
          self._name = name
      
      # ✅ 正确的顺序
      @property                    # 1. property装饰器
      def name(self):
          return self._name
      
      @name.setter                 # 2. setter装饰器（紧跟在property之后）
      @validate_name               # 3. 自定义验证装饰器
      def name(self, value):
          if len(value) < 2:
              raise ValueError("姓名至少2个字符")
          self._name = value
      
      @name.deleter                # 4. deleter装饰器
      @confirm_deletion            # 5. 确认删除装饰器
      def name(self):
          print("删除姓名")
          del self._name
  ```

- 异步装饰器顺序规范
  
  ```
  # ✅ 异步装饰器的正确顺序
  class AsyncAPI:
      # 从外到内的正确顺序：
      # 1. 类装饰器（如 @classmethod, @staticmethod）
      # 2. 重试装饰器
      # 3. 缓存装饰器  
      # 4. 监控/日志装饰器
      # 5. 参数验证装饰器
      
      @classmethod
      @async_retry(max_attempts=3)      # 2. 重试（先重试再缓存）
      @async_cache(ttl=60)              # 3. 缓存
      @async_timer                      # 4. 计时监控
      async def fetch_data(cls, url: str) -> dict:
          """5. 核心业务逻辑"""
          # 模拟异步HTTP请求
          await asyncio.sleep(0.5)
          return {"url": url, "data": "some data"}
  ```

- 装饰器顺序的通用规则
  
  ```
  # 装饰器顺序的通用规则（从最外层到最内层）：
  """
  1. 类装饰器 (@classmethod, @staticmethod, @property)
  2. 路由装饰器 (@app.route, @router.get 等)
  3. 认证/授权装饰器 (@login_required, @permission_required)
  4. 限流/频率限制装饰器
  5. 缓存装饰器 (@cache, @lru_cache)
  6. 重试/容错装饰器
  7. 日志/监控装饰器 (@log_execution, @track_performance)
  8. 参数验证/类型检查装饰器
  9. 数据库事务装饰器 (@transaction.atomic)
  10. 核心业务逻辑（函数本身）
  """
  
  # 记忆口诀：类路认证限缓存，重试日志验事务，最后才是真业务
  
  # 实际应用中的复合装饰器
  def create_api_endpoint(func):
      """组合多个装饰器，确保正确顺序"""
      @wraps(func)
      def decorated_function(*args, **kwargs):
          # 这个装饰器内部按正确顺序组合了其他装饰器的功能
          return func(*args, **kwargs)
      return decorated_function
  
  # 使用工厂函数创建正确顺序的装饰器
  def api_endpoint(rate_limit=100, require_auth=True, cache_ttl=300):
      """装饰器工厂，确保正确的装饰器顺序"""
      def decorator(func):
          # 按正确顺序应用装饰器
          if require_auth:
              func = require_auth_decorator(func)
          
          func = rate_limit_decorator(rate_limit)(func)
          
          if cache_ttl:
              func = cache_decorator(ttl=cache_ttl)(func)
          
          func = log_decorator(func)
          func = validate_args_decorator(func)
          
          return func
      return decorator
  
  # 使用
  @app.route('/api/data')
  @api_endpoint(rate_limit=50, require_auth=True, cache_ttl=600)
  def get_data():
      """核心业务逻辑 - 装饰器顺序由 api_endpoint 工厂保证"""
      return {"data": "some data"}
  ```
  
  1. 类装饰器 (@classmethod, @staticmethod, @property)
  2. 路由装饰器 (@app.route, @router.get 等)
  3. 认证/授权装饰器 (@login_required, @permission_required)
  4. 限流/频率限制装饰器
  5. 缓存装饰器 (@cache, @lru_cache)
  6. 重试/容错装饰器
  7. 日志/监控装饰器 (@log_execution, @track_performance)
  8. 参数验证/类型检查装饰器
  9. 数据库事务装饰器 (@transaction.atomic)
  10. 核心业务逻辑（函数本身）

## 3、文档与注释

### 3.1 文档字符串（Docstrings）

- 模块、类、公有函数必须有文档字符串
- 使用三引号格式，符合PEP 257规范
- 包含功能说明、参数、返回值和异常
- 使用类型注解增强可读性

### 3.2 注释规范

- 行注释使用`#`，与代码至少间隔两个空格
- 注释解释"为什么"而不是"做什么"
- 及时更新过时的注释
- 复杂算法需有详细注释

### 3.2 代码文件
- 代码名需要见名知意
- 单个文件代码行数不超过500行
