import numpy as np
from pathlib import Path
import argparse, os, tqdm

def generate_iq(num_samples: int, snr_db: float, modulation: str):
 
    modulation = modulation.lower()

    if modulation == "bpsk":
        bits = np.random.randint(0, 2, num_samples)
        symbols = 1 - 2*bits
        symbols = symbols.astype(np.complex64)
        # Already unit power

    elif modulation == "qpsk":
        bits = np.random.randint(0, 2, (num_samples, 2))
        mapping = {
            (0,0):  1+1j,
            (0,1):  1-1j,
            (1,1): -1-1j,
            (1,0): -1+1j
        }
        symbols = np.array([mapping[tuple(b)] for b in bits], dtype=np.complex64)
        symbols /= np.sqrt(2)  # Normalize avg power to 1

    elif modulation == "qam16":
        bits = np.random.randint(0, 2, (num_samples, 4))

        def map_16qam(b):
            i = (1 - 2*b[0]) * (2 - b[1])
            q = (1 - 2*b[2]) * (2 - b[3])
            return i + 1j*q

        symbols = np.array([map_16qam(b) for b in bits], dtype=np.complex64)
        symbols /= np.sqrt(10)  # Normalize avg power to 1

    elif modulation == "qam64":
        bits = np.random.randint(0, 2, (num_samples, 6))

        # Map 3 Gray-coded bits to amplitude ∈ {±1,±3,±5,±7}
        def gray_map_3bits(b3):
            g = b3[0]*4 + b3[1]*2 + b3[2]
            gray_order = [0,1,3,2,6,7,5,4]
            level = gray_order[g]
            return 2*level - 7

        symbols = np.array([
            gray_map_3bits(b[:3]) + 1j*gray_map_3bits(b[3:])
            for b in bits
        ], dtype=np.complex64)

        symbols /= np.sqrt(42)  # Normalize avg power to 1

    else:
        raise ValueError("Unsupported modulation. Choose: bpsk, qpsk, qam16, qam64")


    snr_linear = 10**(snr_db/10)
    Ps = 1  # We normalized all to unit power
    Pn = Ps / snr_linear

    noise = np.sqrt(Pn/2) * (np.random.randn(num_samples) + 1j*np.random.randn(num_samples))

    iq_noisy = symbols + noise

    return iq_noisy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=str, required=True, help='Output file path for the generated data')
    p.add_argument('--num', type=int, default=1000, help='Number of examples to generate')
    p.add_argument('--mod', type=str, default=[], action='append', help='Modulation schemes (e.g., qpsk, bpsk)')
    p.add_argument('--snr_db', type=float, default=20.0, help='Signal-to-noise ratio in dB')
    p.add_argument('--samples', type=int, default=1024, help='Number of samples generated per example')
    args = p.parse_args()

    mod_schemes = args.mod if args.mod else ['bpsk', 'qpsk', 'qam16', 'qam64']
    num_examples = args.num
    snr_db = args.snr_db
    num_samples = args.samples
    output_path = Path(args.output)

    os.umask(0)
    os.makedirs(Path(output_path, 'signals'), mode=0o777)
    os.makedirs(Path(output_path, 'labels'), mode=0o777)

    for example in tqdm.tqdm(range(num_examples)):        
        for label, mod_scheme in enumerate(mod_schemes):
            buf = generate_iq(num_samples, snr_db, mod_scheme)
            
            filebase = 'example_%.3d_%s' % (example, mod_scheme)
            np.save(Path(output_path, 'signals', filebase + '.npy'), buf)
            with open(Path(output_path, 'labels', filebase + '.txt'), 'w') as f:
                f.write(str(label))
        
if __name__ == '__main__':
    main()
