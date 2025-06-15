import json
import ast
from datetime import datetime

class CodeAnalyzerTestSuite:
    """Test suite for code analyzer agent functionality"""
    
    def __init__(self):
        self.test_code_samples = {
            "two_sum_hash_map": '''
def two_sum(nums, target):
    num_dict = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_dict:
            return [num_dict[complement], i]
        num_dict[num] = i
    return []
''',
            "two_sum_brute_force": '''
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
''',
            "incomplete_solution": '''
def two_sum(nums, target):
    # TODO: implement this
    pass
''',
            "syntax_error_code": '''
def two_sum(nums, target)
    for i in range(len(nums))
        if nums[i] + nums[j] == target
            return [i, j]
''',
            "binary_search": '''
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
''',
            "dynamic_programming": '''
def fibonacci(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
''',
            "advanced_solution": '''
from collections import defaultdict

def group_anagrams(strs):
    anagram_map = defaultdict(list)
    
    for s in strs:
        # Sort the string to create a key
        key = ''.join(sorted(s))
        anagram_map[key].append(s)
    
    return list(anagram_map.values())
'''
        }
    
    def test_ast_parsing(self):
        """Test AST parsing functionality"""
        print("🔍 Testing AST Parsing")
        print("=" * 40)
        
        for name, code in self.test_code_samples.items():
            print(f"\nTesting: {name}")
            try:
                tree = ast.parse(code)
                
                functions = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions.append({
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "line": node.lineno
                        })
                
                print(f"✅ Functions found: {functions}")
                print(f"   Total AST nodes: {len(list(ast.walk(tree)))}")
                
            except SyntaxError as e:
                print(f"❌ Syntax Error: {e}")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def test_pattern_detection(self):
        """Test algorithmic pattern detection"""
        print("\n🎯 Testing Pattern Detection")
        print("=" * 40)
        
        # Pattern keywords for testing
        patterns = {
            "hash_map": ["dict", "{}}", "in dict", "complement"],
            "two_pointers": ["left", "right", "while left <= right"],
            "binary_search": ["left", "right", "mid", "// 2"],
            "dynamic_programming": ["dp", "dp[", "for i in range"],
            "brute_force": ["for i", "for j", "range(i + 1"]
        }
        
        expected_patterns = {
            "two_sum_hash_map": ["hash_map"],
            "two_sum_brute_force": ["brute_force"],
            "binary_search": ["binary_search", "two_pointers"],
            "dynamic_programming": ["dynamic_programming"],
            "advanced_solution": ["hash_map"]
        }
        
        for code_name, code in self.test_code_samples.items():
            if code_name in expected_patterns:
                print(f"\nAnalyzing: {code_name}")
                
                detected = []
                for pattern_name, keywords in patterns.items():
                    matches = sum(1 for keyword in keywords if keyword in code.lower())
                    if matches > 0:
                        confidence = matches / len(keywords)
                        detected.append((pattern_name, confidence))
                
                detected.sort(key=lambda x: x[1], reverse=True)
                
                print(f"Expected: {expected_patterns[code_name]}")
                print(f"Detected: {[p[0] for p in detected[:2]]}")
                
                # Check if we detected expected patterns
                detected_names = [p[0] for p in detected]
                expected = expected_patterns[code_name]
                success = any(exp in detected_names for exp in expected)
                print(f"✅ Success: {success}")
    
    def test_complexity_analysis(self):
        """Test time/space complexity analysis"""
        print("\n⏱️ Testing Complexity Analysis")
        print("=" * 40)
        
        expected_complexity = {
            "two_sum_hash_map": {"time": "O(n)", "space": "O(n)"},
            "two_sum_brute_force": {"time": "O(n²)", "space": "O(1)"},
            "binary_search": {"time": "O(log n)", "space": "O(1)"},
            "dynamic_programming": {"time": "O(n)", "space": "O(n)"},
            "advanced_solution": {"time": "O(n)", "space": "O(n)"}
        }
        
        for code_name, code in self.test_code_samples.items():
            if code_name in expected_complexity:
                print(f"\nAnalyzing: {code_name}")
                
                try:
                    tree = ast.parse(code)
                    
                    # Count nested loops
                    loop_depth = 0
                    max_depth = 0
                    current_depth = 0
                    
                    class LoopCounter(ast.NodeVisitor):
                        def __init__(self):
                            self.max_depth = 0
                            self.current_depth = 0
                        
                        def visit_For(self, node):
                            self.current_depth += 1
                            self.max_depth = max(self.max_depth, self.current_depth)
                            self.generic_visit(node)
                            self.current_depth -= 1
                        
                        def visit_While(self, node):
                            self.current_depth += 1
                            self.max_depth = max(self.max_depth, self.current_depth)
                            self.generic_visit(node)
                            self.current_depth -= 1
                    
                    counter = LoopCounter()
                    counter.visit(tree)
                    
                    # Estimate complexity
                    if counter.max_depth == 0:
                        estimated_time = "O(1)"
                    elif counter.max_depth == 1:
                        estimated_time = "O(n)"
                    elif counter.max_depth == 2:
                        estimated_time = "O(n²)"
                    else:
                        estimated_time = f"O(n^{counter.max_depth})"
                    
                    # Check for specific patterns
                    if "// 2" in code and "while" in code:
                        estimated_time = "O(log n)"
                    
                    # Estimate space
                    if "dict" in code or "defaultdict" in code or "dp" in code:
                        estimated_space = "O(n)"
                    else:
                        estimated_space = "O(1)"
                    
                    expected = expected_complexity[code_name]
                    print(f"Expected: {expected['time']} time, {expected['space']} space")
                    print(f"Estimated: {estimated_time} time, {estimated_space} space")
                    
                    time_match = estimated_time == expected['time']
                    space_match = estimated_space == expected['space']
                    print(f"✅ Time complexity correct: {time_match}")
                    print(f"✅ Space complexity correct: {space_match}")
                    
                except Exception as e:
                    print(f"❌ Analysis error: {e}")
    
    def test_syntax_checking(self):
        """Test syntax error detection"""
        print("\n🔍 Testing Syntax Checking")
        print("=" * 40)
        
        syntax_tests = {
            "valid_code": ("two_sum_hash_map", True),
            "syntax_error": ("syntax_error_code", False),
            "incomplete_code": ("incomplete_solution", True)  # Syntactically valid but incomplete
        }
        
        for test_name, (code_key, should_be_valid) in syntax_tests.items():
            print(f"\nTesting: {test_name}")
            code = self.test_code_samples[code_key]
            
            try:
                ast.parse(code)
                is_valid = True
                error_msg = None
            except SyntaxError as e:
                is_valid = False
                error_msg = str(e)
            
            print(f"Expected valid: {should_be_valid}")
            print(f"Actually valid: {is_valid}")
            print(f"✅ Correct detection: {is_valid == should_be_valid}")
            
            if error_msg:
                print(f"Error: {error_msg}")
    
    def test_completeness_assessment(self):
        """Test code completeness assessment"""
        print("\n📊 Testing Completeness Assessment")
        print("=" * 40)
        
        completeness_tests = {
            "two_sum_hash_map": 95,      # Complete solution
            "two_sum_brute_force": 90,   # Complete but inefficient
            "incomplete_solution": 25,    # Just skeleton
            "binary_search": 95,         # Complete solution
            "advanced_solution": 98      # Complete with imports
        }
        
        for code_name, expected_score in completeness_tests.items():
            print(f"\nAssessing: {code_name}")
            code = self.test_code_samples[code_name]
            
            try:
                tree = ast.parse(code)
                
                has_function = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
                has_return = any(isinstance(node, ast.Return) for node in ast.walk(tree))
                has_logic = any(isinstance(node, (ast.If, ast.For, ast.While)) for node in ast.walk(tree))
                
                score = 0
                if has_function: score += 30
                if has_return: score += 25
                if has_logic: score += 25
                
                if 'pass' in code:
                    score = max(0, score - 20)
                
                if len(code.strip()) > 50 and score >= 80:
                    score = min(100, score + 20)
                
                print(f"Expected score: ~{expected_score}")
                print(f"Calculated score: {score}")
                print(f"Difference: {abs(score - expected_score)}")
                print(f"✅ Close estimate: {abs(score - expected_score) <= 15}")
                
            except SyntaxError:
                print(f"Syntax error - score: 10")
    
    def run_all_tests(self):
        """Run all test suites"""
        print("🧪 Code Analyzer Test Suite")
        print("=" * 50)
        
        self.test_ast_parsing()
        self.test_pattern_detection()
        self.test_complexity_analysis()
        self.test_syntax_checking()
        self.test_completeness_assessment()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        print("\n🎯 Key Capabilities Tested:")
        print("  ✓ AST parsing and structural analysis")
        print("  ✓ Algorithmic pattern recognition")
        print("  ✓ Time/space complexity estimation")
        print("  ✓ Syntax error detection")
        print("  ✓ Solution completeness assessment")

def simulate_interview_code_analysis():
    """Simulate how code analyzer would work in real interview"""
    print("\n🎬 Interview Code Analysis Simulation")
    print("=" * 50)
    
    interview_progression = [
        ("Initial skeleton", '''
def two_sum(nums, target):
    # TODO: solve this
    pass
'''),
        ("First attempt", '''
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
'''),
        ("Optimized solution", '''
def two_sum(nums, target):
    num_dict = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_dict:
            return [num_dict[complement], i]
        num_dict[num] = i
    return []
''')
    ]
    
    for stage, code in interview_progression:
        print(f"\n📝 {stage}:")
        print("Code submitted...")
        
        # Simulate analysis
        try:
            tree = ast.parse(code)
            has_function = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
            has_return = any(isinstance(node, ast.Return) for node in ast.walk(tree))
            has_logic = any(isinstance(node, (ast.If, ast.For, ast.While)) for node in ast.walk(tree))
            
            # Complexity estimation
            loop_count = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
            if loop_count == 0:
                complexity = "O(1)"
            elif loop_count == 1:
                complexity = "O(n)"
            else:
                complexity = "O(n²)"
            
            # Pattern detection
            if "dict" in code:
                pattern = "Hash Map"
            elif loop_count >= 2:
                pattern = "Brute Force"
            else:
                pattern = "Unknown"
            
            print(f"🔍 Analysis: {pattern} approach, {complexity} complexity")
            print(f"📊 Structure: Function={has_function}, Return={has_return}, Logic={has_logic}")
            
            if "pass" in code:
                print("💭 Feedback: Good start! Try implementing the logic.")
            elif loop_count >= 2:
                print("💭 Feedback: Working solution! Can you optimize the time complexity?")
            elif "dict" in code:
                print("💭 Feedback: Excellent optimization! Great use of hash map.")
                
        except SyntaxError as e:
            print(f"❌ Syntax Error: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    test_suite = CodeAnalyzerTestSuite()
    test_suite.run_all_tests()
    simulate_interview_code_analysis()