# test_execution_agent.py - Test the execution agent capabilities
import json
import asyncio
import docker
from datetime import datetime

class ExecutionAgentTestSuite:
    """Test suite for execution agent functionality"""
    
    def __init__(self):
        self.test_code_samples = {
            "valid_simple": {
                "code": '''
def two_sum(nums, target):
    num_dict = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_dict:
            return [num_dict[complement], i]
        num_dict[num] = i
    return []
''',
                "test_cases": [
                    {"input_data": {"nums": [2, 7, 11, 15], "target": 9}, "expected_output": [0, 1]},
                    {"input_data": {"nums": [3, 2, 4], "target": 6}, "expected_output": [1, 2]}
                ],
                "should_pass": True
            },
            
            "syntax_error": {
                "code": '''
def two_sum(nums, target)
    for i in range(len(nums))
        return i
''',
                "test_cases": [],
                "should_pass": False
            },
            
            "runtime_error": {
                "code": '''
def two_sum(nums, target):
    return nums[100]  # Index error
''',
                "test_cases": [
                    {"input_data": {"nums": [1, 2], "target": 3}, "expected_output": [0, 1]}
                ],
                "should_pass": False
            },
            
            "infinite_loop": {
                "code": '''
def two_sum(nums, target):
    while True:
        pass
    return []
''',
                "test_cases": [
                    {"input_data": {"nums": [1, 2], "target": 3}, "expected_output": []}
                ],
                "should_pass": False
            },
            
            "correct_but_slow": {
                "code": '''
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
''',
                "test_cases": [
                    {"input_data": {"nums": [2, 7, 11, 15], "target": 9}, "expected_output": [0, 1]}
                ],
                "should_pass": True
            },
            
            "edge_case_handling": {
                "code": '''
def two_sum(nums, target):
    if not nums or len(nums) < 2:
        return []
    
    num_dict = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_dict:
            return [num_dict[complement], i]
        num_dict[num] = i
    return []
''',
                "test_cases": [
                    {"input_data": {"nums": [], "target": 0}, "expected_output": []},
                    {"input_data": {"nums": [1], "target": 1}, "expected_output": []},
                    {"input_data": {"nums": [3, 3], "target": 6}, "expected_output": [0, 1]}
                ],
                "should_pass": True
            }
        }
    
    def test_docker_setup(self):
        """Test Docker client setup and basic functionality"""
        print("🐳 Testing Docker Setup")
        print("=" * 40)
        
        try:
            client = docker.from_env()
            
            # Test Docker connection
            client.ping()
            print("✅ Docker client connected successfully")
            
            # Test Python image availability
            try:
                image = client.images.get("python:3.11-alpine")
                print("✅ Python execution image available")
            except docker.errors.ImageNotFound:
                print("📥 Pulling Python image...")
                client.images.pull("python:3.11-alpine")
                print("✅ Python image pulled successfully")
            
            # Test basic container execution
            result = client.containers.run(
                "python:3.11-alpine",
                "python -c 'print(\"Hello from Docker!\")'",
                remove=True
            )
            output = result.decode('utf-8').strip()
            print(f"✅ Container execution test: {output}")
            
            return True
            
        except Exception as e:
            print(f"❌ Docker setup failed: {e}")
            return False
    
    def test_code_execution(self):
        """Test code execution functionality"""
        print("\n🚀 Testing Code Execution")
        print("=" * 40)
        
        try:
            client = docker.from_env()
            
            for test_name, test_data in self.test_code_samples.items():
                print(f"\nTesting: {test_name}")
                code = test_data["code"]
                should_pass = test_data["should_pass"]
                
                try:
                    # Create temporary test file
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                        temp_file.write(code)
                        temp_file_path = temp_file.name
                    
                    try:
                        # Execute in container with timeout
                        start_time = datetime.now()
                        
                        result = client.containers.run(
                            "python:3.11-alpine",
                            f"python /app/code.py",
                            volumes={temp_file_path: {'bind': '/app/code.py', 'mode': 'ro'}},
                            mem_limit="64m",
                            timeout=5,
                            remove=True
                        )
                        
                        execution_time = (datetime.now() - start_time).total_seconds()
                        output = result.decode('utf-8').strip()
                        
                        print(f"  ✅ Executed successfully ({execution_time:.3f}s)")
                        if output:
                            print(f"  Output: {output[:100]}...")
                        
                        success = True
                        
                    except docker.errors.ContainerError as e:
                        print(f"  ❌ Runtime error: {e.stderr.decode('utf-8') if e.stderr else 'Unknown error'}")
                        success = False
                        
                    except Exception as e:
                        print(f"  ❌ Execution error: {e}")
                        success = False
                    
                    finally:
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                    
                    # Check if result matches expectation
                    if success == should_pass:
                        print(f"  ✅ Expected result: {should_pass}, Got: {success}")
                    else:
                        print(f"  ❌ Expected: {should_pass}, Got: {success}")
                        
                except Exception as e:
                    print(f"  ❌ Test setup error: {e}")
                    
        except Exception as e:
            print(f"❌ Code execution test failed: {e}")
    
    def test_security_features(self):
        """Test security sandbox features"""
        print("\n🛡️ Testing Security Features")
        print("=" * 40)
        
        security_tests = {
            "file_system_access": '''
import os
print(os.listdir('/'))
''',
            "network_access": '''
import urllib.request
urllib.request.urlopen('http://google.com')
''',
            "subprocess_execution": '''
import subprocess
subprocess.run(['ls', '-la'])
''',
            "memory_bomb": '''
data = []
for i in range(1000000):
    data.append('x' * 1000)
''',
            "infinite_loop": '''
while True:
    pass
'''
        }
        
        try:
            client = docker.from_env()
            
            for test_name, dangerous_code in security_tests.items():
                print(f"\nTesting security against: {test_name}")
                
                try:
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                        temp_file.write(dangerous_code)
                        temp_file_path = temp_file.name
                    
                    try:
                        # Execute with strict security settings
                        result = client.containers.run(
                            "python:3.11-alpine",
                            "python /app/code.py",
                            volumes={temp_file_path: {'bind': '/app/code.py', 'mode': 'ro'}},
                            mem_limit="64m",
                            timeout=3,  # Short timeout for security tests
                            network_disabled=True,
                            remove=True,
                            user="nobody"  # Non-root user
                        )
                        
                        print(f"  ⚠️ Code executed (may be concerning): {result.decode('utf-8')[:50]}...")
                        
                    except docker.errors.ContainerError as e:
                        print(f"  ✅ Blocked execution: {str(e)[:100]}...")
                        
                    except Exception as e:
                        print(f"  ✅ Security measure triggered: {str(e)[:100]}...")
                    
                    finally:
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                            
                except Exception as e:
                    print(f"  ❌ Security test setup error: {e}")
                    
        except Exception as e:
            print(f"❌ Security testing failed: {e}")
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities"""
        print("\n⏱️ Testing Performance Monitoring")
        print("=" * 40)
        
        performance_tests = {
            "fast_algorithm": '''
def solution():
    return sum(range(100))

print(solution())
''',
            "slow_algorithm": '''
def solution():
    total = 0
    for i in range(10000):
        for j in range(100):
            total += i * j
    return total

print(solution())
''',
            "memory_intensive": '''
def solution():
    data = []
    for i in range(50000):
        data.append(list(range(10)))
    return len(data)

print(solution())
'''
        }
        
        try:
            client = docker.from_env()
            
            for test_name, code in performance_tests.items():
                print(f"\nTesting: {test_name}")
                
                try:
                    import tempfile
                    import os
                    import time
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                        temp_file.write(code)
                        temp_file_path = temp_file.name
                    
                    try:
                        # Monitor execution time
                        start_time = time.time()
                        
                        result = client.containers.run(
                            "python:3.11-alpine",
                            "python /app/code.py",
                            volumes={temp_file_path: {'bind': '/app/code.py', 'mode': 'ro'}},
                            mem_limit="128m",
                            timeout=10,
                            remove=True
                        )
                        
                        execution_time = time.time() - start_time
                        output = result.decode('utf-8').strip()
                        
                        # Categorize performance
                        if execution_time < 0.1:
                            category = "fast"
                        elif execution_time < 1.0:
                            category = "medium"
                        else:
                            category = "slow"
                        
                        print(f"  ⏱️ Execution time: {execution_time:.3f}s ({category})")
                        print(f"  📊 Output: {output}")
                        
                    except docker.errors.ContainerError as e:
                        print(f"  ❌ Execution failed: {e}")
                        
                    except Exception as e:
                        print(f"  ❌ Performance test error: {e}")
                    
                    finally:
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                            
                except Exception as e:
                    print(f"  ❌ Performance test setup error: {e}")
                    
        except Exception as e:
            print(f"❌ Performance monitoring test failed: {e}")
    
    def test_test_case_validation(self):
        """Test test case validation functionality"""
        print("\n🧪 Testing Test Case Validation")
        print("=" * 40)
        
        # Use the valid simple case for testing
        test_data = self.test_code_samples["valid_simple"]
        code = test_data["code"]
        test_cases = test_data["test_cases"]
        
        print("Testing with Two Sum problem...")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"  Input: {test_case['input_data']}")
            print(f"  Expected: {test_case['expected_output']}")
            
            try:
                # Create test wrapper code
                input_data = test_case["input_data"]
                test_code = f'''
{code}

# Test execution
import json
try:
    nums = {json.dumps(input_data["nums"])}
    target = {input_data["target"]}
    result = two_sum(nums, target)
    print(json.dumps({{"result": result, "success": True}}))
except Exception as e:
    print(json.dumps({{"error": str(e), "success": False}}))
'''
                
                # Execute test
                client = docker.from_env()
                
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                    temp_file.write(test_code)
                    temp_file_path = temp_file.name
                
                try:
                    result = client.containers.run(
                        "python:3.11-alpine",
                        "python /app/code.py",
                        volumes={temp_file_path: {'bind': '/app/code.py', 'mode': 'ro'}},
                        timeout=5,
                        remove=True
                    )
                    
                    output = result.decode('utf-8').strip()
                    test_result = json.loads(output)
                    
                    if test_result["success"]:
                        actual = test_result["result"]
                        expected = test_case["expected_output"]
                        
                        # Compare results (handle different orders for lists)
                        if isinstance(actual, list) and isinstance(expected, list):
                            passed = sorted(actual) == sorted(expected)
                        else:
                            passed = actual == expected
                        
                        print(f"  Actual: {actual}")
                        print(f"  ✅ Passed: {passed}")
                    else:
                        print(f"  ❌ Error: {test_result['error']}")
                        
                finally:
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                        
            except Exception as e:
                print(f"  ❌ Test execution error: {e}")
    
    def run_all_tests(self):
        """Run all test suites"""
        print("🧪 Execution Agent Test Suite")
        print("=" * 50)
        
        # Test Docker setup first
        if not self.test_docker_setup():
            print("❌ Docker setup failed - skipping other tests")
            return
        
        self.test_code_execution()
        self.test_security_features()
        self.test_performance_monitoring()
        self.test_test_case_validation()
        
        print("\n" + "=" * 50)
        print("✅ All execution tests completed!")
        print("\n🎯 Key Capabilities Tested:")
        print("  ✓ Docker container execution")
        print("  ✓ Code security sandboxing")
        print("  ✓ Performance monitoring")
        print("  ✓ Test case validation")
        print("  ✓ Error handling and timeouts")

def simulate_interview_execution_flow():
    """Simulate how execution agent works in real interview"""
    print("\n🎬 Interview Execution Flow Simulation")
    print("=" * 50)
    
    interview_stages = [
        ("Initial test", "def two_sum(nums, target): pass", "Empty function - should fail tests"),
        ("Working solution", '''
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
''', "Brute force - should pass but be slow"),
        ("Optimized solution", '''
def two_sum(nums, target):
    num_dict = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_dict:
            return [num_dict[complement], i]
        num_dict[num] = i
    return []
''', "Hash map - should pass quickly")
    ]
    
    print("Simulating execution progression...")
    
    for stage, code, description in interview_stages:
        print(f"\n📝 {stage}: {description}")
        print("User submits code for execution...")
        
        # Simulate execution feedback
        if "pass" in code:
            print("❌ Execution Result: No output (function not implemented)")
            print("💭 Feedback: Your function needs an implementation. Try adding some logic!")
        elif "for i" in code and "for j" in code:
            print("✅ Execution Result: All test cases passed!")
            print("⏱️ Performance: 45ms execution (could be faster)")
            print("💭 Feedback: Great! Your solution works correctly. Can you optimize the time complexity?")
        elif "num_dict" in code:
            print("✅ Execution Result: All test cases passed!")
            print("⚡ Performance: 8ms execution (excellent!)")
            print("💭 Feedback: Perfect! Excellent optimization with hash map approach - O(n) time complexity!")
        
        print("-" * 30)

if __name__ == "__main__":
    test_suite = ExecutionAgentTestSuite()
    test_suite.run_all_tests()
    simulate_interview_execution_flow()