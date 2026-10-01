import itertools
import subprocess
import sys
import argparse

def main():
    # 1. Parse manual conditions for this batch run
    parser = argparse.ArgumentParser(description="Automated FL/HFL Matrix runner for a fixed Alpha.")
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value for non-IID distribution')
    batch_args = parser.parse_args()

    # 2. Define experimental dimensions
    # skews = ['label', 'feature', 'quantity', 'label_feature', 'label_quantity', 'feature_quantity', 'label_feature_quantity']
    skews = ['label']

    learning_methods = ['fl', 'hfl']
    # learning_methods = ['hfl']

    # update_methods = ['fedavg', 'fedprox', 'scaffold','fednova', 'feddf', 'moon']
    update_methods = ['moon']

    # propagation_losses = ['normal', 'fspl', 'nfspl']
    propagation_losses = ['normal']

    # 3. Build the cross-product combinations
    base_combinations = list(itertools.product(skews, learning_methods, update_methods, propagation_losses))

    # 4. Apply the KPCA constraint (KPCA runs on fspl and nfspl only)
    final_runs = []
    for skew, lm, um, prop_loss in base_combinations:
        if prop_loss == 'normal':
            final_runs.append((skew, lm, um, prop_loss, False))
        else:
            final_runs.append((skew, lm, um, prop_loss, False))
            final_runs.append((skew, lm, um, prop_loss, True))

    total_runs = len(final_runs)
    print("\n" + "="*60)
    print(f"📊 BATCH SIMULATION INITIALIZED")
    print(f" Fixed Alpha Value     : {batch_args.alpha}")
    print(f" Total Runs in Batch   : {total_runs} combinations")
    print("="*60 + "\n")

    # 5. Sequentially execute main.py over the reduced grid loop
    for idx, (skew, lm, um, prop_loss, use_kpca) in enumerate(final_runs, 1):
        print(f"🔄 Run [{idx}/{total_runs}] -> Skew: {skew.upper()} | LM: {lm.upper()} | Strategy: {um.upper()} | Channel: {prop_loss.upper()} | KPCA: {use_kpca}")
        
        # Assemble CLI argument list
        cmd = [
            sys.executable, 'main.py',
            '--dirichlet', str(skew),  # Update this to '--skew' if your main.py expects that instead
            '--alpha', str(batch_args.alpha),
            '--learning_method', str(lm),
            '--update_method', str(um),
            '--propagation_loss', str(prop_loss)
        ]
        
        # Append KPCA flag if active
        if use_kpca:
            cmd.extend(['--kpca', 'True'])
        else:
            cmd.extend(['--kpca', 'False'])
            
        try:
            # Block and run process
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ [ERROR] Run {idx} failed with exit code {e.returncode}. Skipping to next configuration...")
        except KeyboardInterrupt:
            print("\n🛑 Execution paused manually by user interrupt. Exiting script safely.")
            sys.exit(0)

    print(f"\n🎉 Batch completed successfully for Alpha: {batch_args.alpha}!")

if __name__ == '__main__':
    main()