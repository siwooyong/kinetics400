import subprocess

if __name__ == "__main__":
    subprocess.run(['tar', '-xzvf', 'data/Kinetics-400.tar.gz', '-C', 'data/'], check = True)