import os
import wave
import random

wav_path = r'an4\\train\\an4\\wav\\'
txt_path = r'an4\\train\\an4\\txt\\'
longest_num_frames = 102400  # calculated using find_longest_file()


def main():
    # List all files in the folder
    wav_files = [file for file in os.listdir(wav_path) if file.endswith('.wav')]

    i = 0

    while i < 4000:
        file1_path, file2_path = random.sample(wav_files, 2)

        # Open the first WAV file
        wav_file1 = wave.open(wav_path + file1_path, 'rb')
        num_frames1 = wav_file1.getnframes()
        params1 = wav_file1.getparams()

        # Open the second WAV file
        wav_file2 = wave.open(wav_path + file2_path, 'rb')
        num_frames2 = wav_file2.getnframes()
        params2 = wav_file2.getparams()

        # Check if the combined num_frames is smaller than the maximum size
        if num_frames1 + num_frames2 < longest_num_frames:
            i += 1
            #print(f"combining \"{file1_path}\" and \"{file2_path}\"")

            # Create a new combined WAV file
            output_path1 = file1_path[:-4] + '_' + file2_path
            with wave.open(wav_path + output_path1, 'wb') as combined_wav:
                combined_wav.setparams(params1)  # Use parameters from the first file
                combined_wav.writeframes(wav_file1.readframes(num_frames1))
                combined_wav.writeframes(wav_file2.readframes(num_frames2))

            # Reset the file pointers of wav_file1 and wav_file2 to the beginning
            wav_file1.rewind()
            wav_file2.rewind()

            # Create a new combined WAV file
            output_path2 = file2_path[:-4] + '_' + file1_path
            with wave.open(wav_path + output_path2, 'wb') as combined_wav:
                combined_wav.setparams(params1)  # Use parameters from the first file
                combined_wav.writeframes(wav_file2.readframes(num_frames2))
                combined_wav.writeframes(wav_file1.readframes(num_frames1))

            # create a new combined txt file
            file1_path, file2_path = file1_path[:-4] + '.txt', file2_path[:-4] + '.txt'
            with open(txt_path + file1_path, 'r') as file1, open(txt_path + file2_path, 'r') as file2:
                data1 = file1.read()
                data2 = file2.read()

            with open(txt_path + file1_path[:-4] + '_' + file2_path, 'w') as destination_file:
                destination_file.write(data1 + ' ' + data2)

            with open(txt_path + file2_path[:-4] + '_' + file1_path, 'w') as destination_file:
                destination_file.write(data2 + ' ' + data1)

        # close files
        wav_file1.close()
        wav_file2.close()

        if i%100 == 0:
            print (i)


def find_longest_file():
    # List all files in the folder
    wav_files = [file for file in os.listdir(wav_path) if file.endswith('.wav')]

    biggest = 0
    file = ''

    # Loop through each WAV file and get the number of frames
    for wav_file in wav_files:
        wav_file_path = os.path.join(wav_path, wav_file)
        with wave.open(wav_file_path, 'rb') as wav_file_obj:
            num_frames = wav_file_obj.getnframes()
            if num_frames > biggest:
                biggest = num_frames
                file = wav_file
            print(f"WAV File: {wav_file}, Frames: {num_frames}")
    print(f"biggest: {file}, frames: {biggest}")


if __name__ == "__main__":
    main()
