from math import sqrt
import numpy as np
from PIL import Image
import argparse

def compress(input_file, temp_file, N, svd):
    img = Image.open(f"/Comp_math/task3/images/{input_file}.bmp")

    img_array = np.array(img, dtype='float32')

    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    if (svd == "numpy") :
        U1, Sigma1, V1 = np.linalg.svd(r)
        U2, Sigma2, V2 = np.linalg.svd(g)
        U3, Sigma3, V3 = np.linalg.svd(b)
    else:
        print("Not implemented")
        exit(1)

    m, n = img.size

    k = int(max(1, ((-(m+n) + sqrt((m+n)**2 + 4 * m * n / 9 / N)) / 2)))
    if k <= 0:
        print("Can't compress image")
        exit(1)

    new_r = np.diag(Sigma1[:k])
    new_g = np.diag(Sigma2[:k])
    new_b = np.diag(Sigma3[:k])

   # plt.imshow(img)
   # plt.show()
    np.savez(f"/Comp_math/task3/images/{temp_file}.npz", 
             U1=U1[:, :k], Sigma1=new_r, V1=V1[:k, :],
             U2=U2[:, :k], Sigma2=new_g, V2=V2[:k, :],
             U3=U3[:, :k], Sigma3=new_b, V3=V3[:k, :])

def decompress(temp_file, result):
    data = np.load(f"/Comp_math/task3/images/{temp_file}.npz")
    U1 = data["U1"]
    Sigma1 = data["Sigma1"]
    V1 = data["V1"]
    U2 = data["U2"]
    Sigma2 = data["Sigma2"]
    V2 = data["V2"]
    U3 = data["U3"]
    Sigma3 = data["Sigma3"]
    V3 = data["V3"]

    r = np.dot(U1, np.dot(Sigma1, V1))
    g = np.dot(U2, np.dot(Sigma2, V2))
    b = np.dot(U3, np.dot(Sigma3, V3))

    compressed_img_array = np.stack((r, g, b), axis=-1)
    image = Image.fromarray(np.uint8(compressed_img_array))
    image.save(f"/Comp_math/task3/images/{result}.bmp")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compress", "decompress"], required=True, help="Mode of operation: compress "
                                                                                          "or decompress")
    parser.add_argument("--method", choices=["numpy", "simple", "smart"], help="Method of singular value "
                                                                                  "decomposition (only required if "
                                                                                  "mode is compress)")
    parser.add_argument("--compression", type=int, help="Compression factor (only required if mode is compress)")
    parser.add_argument("--in_file", required=True, help="Input file path")
    parser.add_argument("--out_file", required=True, help="Output file path")
    args = parser.parse_args()
    mode = args.mode
    method = args.method
    compression = args.compression
    in_file = args.in_file
    out_file = args.out_file

    if mode == "compress":
        if method is None or compression is None:
            ValueError("Error: --method and --compression are required for compression mode.")
        if method == "numpy":
            compress(in_file, out_file, compression, method)
        elif method == "simple":
            ValueError("Not implemented")
        elif method == "smart":
            ValueError("Not implemented")

    elif mode == "decompress":
        decompress(in_file, out_file)
    else:
        ValueError("Error: Invalid mode. Mode must be 'compress' or 'decompress'.")
