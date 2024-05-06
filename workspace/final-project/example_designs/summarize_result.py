import os
import argparse


stat_file = "timeloop-mapper.stats.txt"

def load_stats(file):

    with open(file, 'r') as file:
        # Read the lines of the file
        lines = file.readlines()

        line_len = len(lines)
        # Loop through the lines to find the line containing "Cycles:"
        for idx in range(line_len):
            line = lines[idx]
            if "Summary Stats" in line:
                cycle_line = lines[idx + 4]
                EDP_line = lines[idx + 6]
                assert "Cycles" in cycle_line
                assert "EDP" in EDP_line
                # Extract the numbers
                cycles = int(cycle_line.split(":")[1].strip())
                EDP = float(EDP_line.split(":")[1].strip())
                energy = EDP / cycles
                break

    return cycles, energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs", type=str, help="path to the logs directory"
    )
    args = parser.parse_args()

    logs = args.logs

    cycles_tot = 0
    energy_tot = 0
    num_layers = 0
    # find all dir under logs
    dirs = [d for d in os.listdir(logs) if os.path.isdir(os.path.join(logs, d))]
    for d in dirs:
        local_path = os.path.join(logs, d)
        files = os.listdir(local_path)
        for f in files:
            if f == stat_file:
                cycles, energy = load_stats(os.path.join(local_path, f))
                cycles_tot += cycles
                energy_tot += energy
                num_layers += 1
    
    print(f"Total Cycles: {cycles_tot}")
    print(f"Total Energy: {energy_tot*1e3} mJ")
    print(f"Total EDP: {energy_tot * cycles_tot} J*cycle")
                



if __name__ == "__main__":
    main()





