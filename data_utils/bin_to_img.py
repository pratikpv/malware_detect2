import multiprocessing
from tqdm import tqdm
import numpy as np
import imageio
import array
import os


def generate_and_save_image(input_filename, output_filename, width):
    f = open(input_filename, 'rb')
    ln = os.path.getsize(input_filename)  # length of file in bytes
    if width == 0:
        width = ln
    rem = ln % width
    a = array.array("B")  # uint8 array
    a.fromfile(f, ln - rem)
    f.close()
    g = np.reshape(a, (len(a) // width, width))
    g = np.uint8(g)
    imageio.imwrite(output_filename, g)  # save the image


def convert_bin_to_img(input_dir, width, max_files=0):
    output_dir = input_dir + '_width_' + str(width)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    list_dirs = os.listdir(input_dir)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:

        jobs = []
        results = []
        total_count = 0

        for dirname in list_dirs:
            list_files = os.listdir(os.path.join(input_dir, dirname))
            count = 0
            for filename in list_files:
                input_filename = os.path.join(input_dir, dirname, filename)
                try:
                    output_filename = os.path.splitext(os.path.basename(input_filename))[0] + '.png'
                    output_class_dir = os.path.join(output_dir, dirname)
                    if not os.path.isdir(output_class_dir):
                        os.mkdir(output_class_dir)
                    output_filename = os.path.join(output_dir, dirname, output_filename)

                    jobs.append(
                        pool.apply_async(generate_and_save_image, (input_filename, output_filename, width)))
                    count += 1
                    if max_files > 0 and max_files == count:
                        break
                except:
                    print('Ignoring ', filename)

            total_count += count
        tqdm_desc = 'Converting Malware bins to images for width ' + str(width)
        for job in tqdm(jobs, desc=tqdm_desc):
            results.append(job.get())
