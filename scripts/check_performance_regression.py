#!/usr/bin/env python3
"""
Performance regression detection script for QuantumEdge CI/CD pipeline.
Compares current benchmark results against baseline performance metrics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class PerformanceRegression:
    """Detect performance regressions in benchmark results."""
    
    def __init__(self, threshold: float = 0.15):
        """
        Initialize regression detector.
        
        Args:
            threshold: Maximum allowed performance degradation (15% default)
        """
        self.threshold = threshold
        self.baseline_file = Path("benchmark-baseline.json")
    
    def load_baseline(self) -> Dict:
        """Load baseline performance metrics."""
        if not self.baseline_file.exists():
            print("⚠️  No baseline found, creating from current results")
            return {}
        
        with open(self.baseline_file) as f:
            return json.load(f)
    
    def analyze_regression(self, current_file: str) -> Tuple[bool, List[str]]:
        """
        Analyze performance regression.
        
        Args:
            current_file: Path to current benchmark results
            
        Returns:
            Tuple of (has_regression, list_of_issues)
        """
        baseline = self.load_baseline()
        
        with open(current_file) as f:
            current = json.load(f)
        
        if not baseline:
            # First run - save as baseline
            with open(self.baseline_file, 'w') as f:
                json.dump(current, f, indent=2)
            print("✅ Baseline established")
            return False, []
        
        issues = []
        has_regression = False
        
        # Check optimization performance
        if "optimization" in current and "optimization" in baseline:
            regression, msgs = self._check_optimization_regression(
                current["optimization"], baseline["optimization"]
            )
            if regression:
                has_regression = True
                issues.extend(msgs)
        
        # Check memory usage
        if "memory" in current and "memory" in baseline:
            regression, msgs = self._check_memory_regression(
                current["memory"], baseline["memory"]
            )
            if regression:
                has_regression = True
                issues.extend(msgs)
        
        # Check throughput
        if "throughput" in current and "throughput" in baseline:
            regression, msgs = self._check_throughput_regression(
                current["throughput"], baseline["throughput"]
            )
            if regression:
                has_regression = True
                issues.extend(msgs)
        
        return has_regression, issues
    
    def _check_optimization_regression(self, current: Dict, baseline: Dict) -> Tuple[bool, List[str]]:
        """Check optimization time regression."""
        issues = []
        has_regression = False
        
        for test_name, current_time in current.items():
            if test_name not in baseline:
                continue
                
            baseline_time = baseline[test_name]
            regression_factor = (current_time - baseline_time) / baseline_time
            
            if regression_factor > self.threshold:
                has_regression = True
                issues.append(
                    f"🚨 {test_name}: {regression_factor:.1%} slower "
                    f"({current_time:.2f}ms vs {baseline_time:.2f}ms)"
                )
            elif regression_factor > 0.05:  # 5% warning threshold
                issues.append(
                    f"⚠️  {test_name}: {regression_factor:.1%} slower "
                    f"({current_time:.2f}ms vs {baseline_time:.2f}ms)"
                )
        
        return has_regression, issues
    
    def _check_memory_regression(self, current: Dict, baseline: Dict) -> Tuple[bool, List[str]]:
        """Check memory usage regression."""
        issues = []
        has_regression = False
        
        for test_name, current_mem in current.items():
            if test_name not in baseline:
                continue
                
            baseline_mem = baseline[test_name]
            regression_factor = (current_mem - baseline_mem) / baseline_mem
            
            if regression_factor > self.threshold:
                has_regression = True
                issues.append(
                    f"🚨 {test_name}: {regression_factor:.1%} more memory "
                    f"({current_mem:.1f}MB vs {baseline_mem:.1f}MB)"
                )
        
        return has_regression, issues
    
    def _check_throughput_regression(self, current: Dict, baseline: Dict) -> Tuple[bool, List[str]]:
        """Check throughput regression."""
        issues = []
        has_regression = False
        
        for test_name, current_tps in current.items():
            if test_name not in baseline:
                continue
                
            baseline_tps = baseline[test_name]
            regression_factor = (baseline_tps - current_tps) / baseline_tps
            
            if regression_factor > self.threshold:
                has_regression = True
                issues.append(
                    f"🚨 {test_name}: {regression_factor:.1%} lower throughput "
                    f"({current_tps:.0f} vs {baseline_tps:.0f} ops/sec)"
                )
        
        return has_regression, issues


def main():
    """Main performance regression check."""
    parser = argparse.ArgumentParser(description="Check for performance regression")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--threshold", type=float, default=0.15, 
                       help="Regression threshold (default: 15%)")
    
    args = parser.parse_args()
    
    detector = PerformanceRegression(threshold=args.threshold)
    has_regression, issues = detector.analyze_regression(args.results_file)
    
    if issues:
        print("\n📊 Performance Analysis Results:")
        print("=" * 40)
        for issue in issues:
            print(issue)
        print()
    
    if has_regression:
        print(f"❌ Performance regression detected (>{args.threshold:.0%} degradation)")
        print("Consider optimizing before merging this change.")
        sys.exit(1)
    else:
        print("✅ No significant performance regression detected")
        sys.exit(0)


if __name__ == "__main__":
    main()