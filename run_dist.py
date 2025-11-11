#!/usr/bin/env python
"""
Distributed Hyperparameter Optimization Runner

Run this script on multiple machines pointing to the same shared directory.
Each machine will automatically pick up uncompleted tasks and coordinate
with other machines to avoid duplicating work.

Usage:
    # On Machine 1:
    python run_distributed.py

    # On Machine 2:
    python run_distributed.py

    # On Machine 3:
    python run_distributed.py

Each machine will find and run different tasks automatically!
"""

import subprocess
import time
import os
import json
import socket
import fcntl
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================

# All datasets and methods to optimize
CONTINUOUS_DATASETS = [
    'swiss_roll',
    's_curve',
    'helix',
    'noisy_moons',
    'tree_3d',
]

CATEGORICAL_DATASETS = [
    'mnist',
    'fashion_mnist',
    # 'gaussian_blobs',
    'varied_gaussian_blobs',
]

METHODS = [
    'tsne',
    'umap',
    'phate',
    'pacmap',
    'mds',
    'autoencoder',  # Comment out to skip (slow!)
]

# Coordination directory (must be shared across all machines!)
COORD_DIR = "./hyperopt_coordination"

# Stale lock timeout (if a lock is older than this, assume machine crashed)
STALE_LOCK_TIMEOUT = 7200  # 2 hours in seconds

# Time to wait between checking for new tasks
POLL_INTERVAL = 5  # seconds

# ============================================================================
# END CONFIGURATION
# ============================================================================


class TaskCoordinator:
    """Coordinates tasks across multiple machines using file-based locking."""
    
    def __init__(self, coord_dir):
        self.coord_dir = Path(coord_dir)
        self.coord_dir.mkdir(exist_ok=True)
        
        self.tasks_dir = self.coord_dir / "tasks"
        self.locks_dir = self.coord_dir / "locks"
        self.completed_dir = self.coord_dir / "completed"
        self.logs_dir = self.coord_dir / "logs"
        
        # Create subdirectories
        self.tasks_dir.mkdir(exist_ok=True)
        self.locks_dir.mkdir(exist_ok=True)
        self.completed_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.machine_id = f"{self.hostname}_{self.pid}"
    
    def initialize_tasks(self, datasets, methods):
        """
        Create task files for all dataset-method combinations.

        Creates two types of tasks:
        1. Optimization tasks (one per dataset-method combination)
        2. Normalization tasks (one per dataset, runs after all methods complete)
        """
        tasks = []

        # Track datasets for normalization tasks
        all_datasets_continuous = datasets.get('continuous', [])
        all_datasets_categorical = datasets.get('categorical', [])

        # Continuous datasets - optimization tasks
        for dataset in all_datasets_continuous:
            for method in methods:
                task = {
                    'dataset': dataset,
                    'data_type': 'continuous',
                    'method': method,
                    'task_type': 'optimize',
                    'task_id': f"continuous_{dataset}_{method}"
                }
                tasks.append(task)

        # Categorical datasets - optimization tasks
        for dataset in all_datasets_categorical:
            for method in methods:
                task = {
                    'dataset': dataset,
                    'data_type': 'categorical',
                    'method': method,
                    'task_type': 'optimize',
                    'task_id': f"categorical_{dataset}_{method}"
                }
                tasks.append(task)

        # Normalization tasks (one per dataset)
        # These run after all methods for a dataset complete
        for dataset in all_datasets_continuous:
            task = {
                'dataset': dataset,
                'data_type': 'continuous',
                'task_type': 'normalize',
                'task_id': f"continuous_{dataset}_normalize",
                'depends_on': [f"continuous_{dataset}_{method}" for method in methods]
            }
            tasks.append(task)

        for dataset in all_datasets_categorical:
            task = {
                'dataset': dataset,
                'data_type': 'categorical',
                'task_type': 'normalize',
                'task_id': f"categorical_{dataset}_normalize",
                'depends_on': [f"categorical_{dataset}_{method}" for method in methods]
            }
            tasks.append(task)

        # Write task files
        for task in tasks:
            task_file = self.tasks_dir / f"{task['task_id']}.json"
            if not task_file.exists():
                with open(task_file, 'w') as f:
                    json.dump(task, f, indent=2)

        return tasks
    
    def is_task_completed(self, task_id):
        """Check if a task has been completed."""
        completed_file = self.completed_dir / f"{task_id}.done"
        return completed_file.exists()
    
    def is_task_locked(self, task_id):
        """Check if a task is currently locked by another machine."""
        lock_file = self.locks_dir / f"{task_id}.lock"
        
        if not lock_file.exists():
            return False
        
        # Check if lock is stale
        try:
            lock_age = time.time() - lock_file.stat().st_mtime
            if lock_age > STALE_LOCK_TIMEOUT:
                print(f"  ⚠️  Stale lock detected for {task_id} (age: {lock_age/3600:.1f}h)")
                # Remove stale lock
                try:
                    lock_file.unlink()
                    print(f"  ✓ Removed stale lock")
                    return False
                except:
                    pass
            
            # Read lock info
            with open(lock_file, 'r') as f:
                lock_info = json.load(f)
            
            # Check if it's our own lock
            if lock_info.get('machine_id') == self.machine_id:
                return False
            
            return True
            
        except Exception as e:
            # If we can't read the lock, assume it's locked
            return True
    
    def acquire_lock(self, task_id):
        """Try to acquire a lock for a task. Returns True if successful."""
        lock_file = self.locks_dir / f"{task_id}.lock"
        
        # Try to create lock file atomically
        try:
            # Use O_CREAT | O_EXCL to ensure atomic creation
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            
            # Write lock info
            lock_info = {
                'machine_id': self.machine_id,
                'hostname': self.hostname,
                'pid': self.pid,
                'timestamp': datetime.now().isoformat(),
                'task_id': task_id
            }
            
            os.write(fd, json.dumps(lock_info, indent=2).encode())
            os.close(fd)
            
            return True
            
        except FileExistsError:
            # Lock already exists
            return False
        except Exception as e:
            print(f"  ✗ Error acquiring lock: {e}")
            return False
    
    def release_lock(self, task_id):
        """Release a lock for a task."""
        lock_file = self.locks_dir / f"{task_id}.lock"
        try:
            lock_file.unlink()
        except:
            pass
    
    def dependencies_satisfied(self, task):
        """Check if all dependencies for a task are satisfied."""
        depends_on = task.get('depends_on', [])

        if not depends_on:
            return True  # No dependencies

        # Check if all dependencies are completed
        for dep_task_id in depends_on:
            if not self.is_task_completed(dep_task_id):
                return False

        return True

    def mark_completed(self, task_id, success, elapsed_time, error=None):
        """Mark a task as completed."""
        completed_file = self.completed_dir / f"{task_id}.done"

        completion_info = {
            'task_id': task_id,
            'machine_id': self.machine_id,
            'hostname': self.hostname,
            'success': success,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat(),
            'error': str(error) if error else None
        }

        with open(completed_file, 'w') as f:
            json.dump(completion_info, f, indent=2)

        # Release lock
        self.release_lock(task_id)
    
    def get_available_task(self):
        """Get the next available task (not completed, not locked, dependencies satisfied)."""
        # Get all task files
        task_files = sorted(self.tasks_dir.glob("*.json"))

        for task_file in task_files:
            task_id = task_file.stem

            # Skip if completed
            if self.is_task_completed(task_id):
                continue

            # Skip if locked
            if self.is_task_locked(task_id):
                continue

            # Load task to check dependencies
            with open(task_file, 'r') as f:
                task = json.load(f)

            # Skip if dependencies not satisfied
            if not self.dependencies_satisfied(task):
                continue

            # Try to acquire lock
            if self.acquire_lock(task_id):
                return task

        return None
    
    def get_progress_summary(self):
        """Get summary of overall progress."""
        all_tasks = list(self.tasks_dir.glob("*.json"))
        completed_tasks = list(self.completed_dir.glob("*.done"))
        locked_tasks = list(self.locks_dir.glob("*.lock"))
        
        total = len(all_tasks)
        completed = len(completed_tasks)
        in_progress = len(locked_tasks)
        remaining = total - completed - in_progress
        
        return {
            'total': total,
            'completed': completed,
            'in_progress': in_progress,
            'remaining': remaining
        }


def run_task(task):
    """Run a single optimization or normalization task."""
    dataset = task['dataset']
    data_type = task['data_type']
    task_type = task.get('task_type', 'optimize')  # 'optimize' or 'normalize'

    if task_type == 'normalize':
        # Normalization task
        cmd = [
            'python', 'normalize_results.py',
            '--dataset', dataset,
            '--data-type', data_type
        ]
    else:
        # Optimization task
        method = task['method']
        cmd = [
            'python', 'hyperopt.py',
            '--dataset', dataset,
            '--data-type', data_type,
            '--methods', method,
            '--skip-normalization'  # Skip normalization in distributed mode
        ]
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time
        
        print(result.stdout)
        
        return True, elapsed, None
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        
        print(f"\n✗ Task failed!")
        print(f"Error: {e}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        
        return False, elapsed, e
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Unexpected error: {e}")
        return False, elapsed, e


def main():
    """Main distributed runner."""
    coordinator = TaskCoordinator(COORD_DIR)
    
    print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print("┃          DISTRIBUTED HYPERPARAMETER OPTIMIZATION                            ┃")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    
    print(f"\nMachine ID: {coordinator.machine_id}")
    print(f"Coordination directory: {COORD_DIR}")
    print(f"Hostname: {coordinator.hostname}")
    print(f"PID: {coordinator.pid}")
    
    # Initialize tasks (safe to call multiple times)
    datasets = {
        'continuous': CONTINUOUS_DATASETS,
        'categorical': CATEGORICAL_DATASETS
    }
    tasks = coordinator.initialize_tasks(datasets, METHODS)
    
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Methods: {', '.join(METHODS)}")
    
    # Show initial progress
    progress = coordinator.get_progress_summary()
    print(f"\nInitial Progress:")
    print(f"  ✓ Completed:   {progress['completed']:3d} / {progress['total']}")
    print(f"  ⏳ In Progress: {progress['in_progress']:3d} / {progress['total']}")
    print(f"  ⏹  Remaining:   {progress['remaining']:3d} / {progress['total']}")
    
    print("\n" + "="*80)
    print("Starting task processing...")
    print("="*80)
    
    tasks_completed_by_me = 0
    start_time = time.time()
    
    while True:
        # Get next available task
        task = coordinator.get_available_task()
        
        if task is None:
            # No tasks available, check if we're done
            progress = coordinator.get_progress_summary()
            
            if progress['remaining'] == 0 and progress['in_progress'] == 0:
                print("\n" + "="*80)
                print("🎉 ALL TASKS COMPLETED!")
                print("="*80)
                break
            
            # Wait and try again
            print(f"\n⏸  No available tasks. Waiting {POLL_INTERVAL}s...")
            print(f"   Status: {progress['completed']} completed, {progress['in_progress']} in progress, {progress['remaining']} remaining")
            time.sleep(POLL_INTERVAL)
            continue
        
        # Run the task
        task_id = task['task_id']
        task_type = task.get('task_type', 'optimize')

        print(f"\n{'='*80}")
        print(f"📋 Picked up task: {task_id}")
        print(f"   Dataset: {task['dataset']} ({task['data_type']})")

        if task_type == 'normalize':
            print(f"   Type: Global Normalization")
        else:
            print(f"   Method: {task['method']}")

        print(f"{'='*80}")
        
        success, elapsed, error = run_task(task)
        
        # Mark as completed
        coordinator.mark_completed(task_id, success, elapsed, error)
        
        if success:
            print(f"\n✓ Task completed successfully in {elapsed/60:.1f} minutes")
            tasks_completed_by_me += 1
        else:
            print(f"\n✗ Task failed after {elapsed/60:.1f} minutes")
        
        # Show progress
        progress = coordinator.get_progress_summary()
        print(f"\nProgress Update:")
        print(f"  ✓ Completed:   {progress['completed']:3d} / {progress['total']}")
        print(f"  ⏳ In Progress: {progress['in_progress']:3d} / {progress['total']}")
        print(f"  ⏹  Remaining:   {progress['remaining']:3d} / {progress['total']}")
        print(f"  📊 My contribution: {tasks_completed_by_me} tasks")
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "┏" + "━"*78 + "┓")
    print("┃" + " "*25 + "FINAL SUMMARY" + " "*40 + "┃")
    print("┗" + "━"*78 + "┛")
    
    print(f"\nMachine: {coordinator.machine_id}")
    print(f"Tasks completed by this machine: {tasks_completed_by_me}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    
    if tasks_completed_by_me > 0:
        avg_time = total_time / tasks_completed_by_me
        print(f"Average time per task: {avg_time/60:.1f} minutes")
    
    # Show completion statistics
    print(f"\nAll Completed Tasks:")
    completed_files = sorted(coordinator.completed_dir.glob("*.done"))
    
    machine_stats = {}
    for cf in completed_files:
        with open(cf, 'r') as f:
            info = json.load(f)
        
        machine_id = info['machine_id']
        if machine_id not in machine_stats:
            machine_stats[machine_id] = {'count': 0, 'time': 0, 'success': 0}
        
        machine_stats[machine_id]['count'] += 1
        machine_stats[machine_id]['time'] += info['elapsed_time']
        if info['success']:
            machine_stats[machine_id]['success'] += 1
    
    print("\nContributions by machine:")
    for machine_id, stats in sorted(machine_stats.items()):
        success_rate = (stats['success'] / stats['count'] * 100) if stats['count'] > 0 else 0
        avg_time = stats['time'] / stats['count'] / 60 if stats['count'] > 0 else 0
        print(f"  {machine_id:30s}: {stats['count']:3d} tasks, "
              f"{success_rate:5.1f}% success, {avg_time:6.1f} min/task")
    
    print(f"\nResults saved to: ./hyperparameter_optimization/")
    print(f"Coordination files: {COORD_DIR}/")
    
    print("\n" + "┏" + "━"*78 + "┓")
    print("┃" + " "*30 + "DONE!" + " "*43 + "┃")
    print("┗" + "━"*78 + "┛\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
        print("Note: Lock files will remain. Other machines can detect and recover.")
        print("Or manually clean: rm -rf ./hyperopt_coordination/locks/*.lock")