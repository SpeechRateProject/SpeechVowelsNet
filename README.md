# Speech Yolo Model - Localize and Detect Vowels - Train and Test

This is part of speech rate measure project

The model was built on the build model of SpeechYolo, by adding output in the last layer
to predict number of vowels in a 500ms audio file

You can see the argumnets  of Speech Yolo Model explained  [here](https://github.com/MLSpeech/speech_yolo)
The model uses object detection methods
Its goal is to localize boundaries of utterances(vowels) within the input signal(500ms of audio file), and classify them
and count how many of them in the input signal,
while its priority is to localize them more then classify them

It composed by a convolution neural network, with loss function

As mentioned, we based on the SpeechYolo project
to create our model we added another output to the last layer
and convert the number of cells to be 32 such that we wont have
 more then one vowel in a cell so we can sum up al the values of 
the cells inside each data file(input signal)
and make the last layer to output also the number of vowels in the input signal,
by sum up all the cells values(every cell value is 1/0)
In addition, we added a loss "six part loss" for the loss function
and save the model by check the mistake value (object/no object)


## Installation instructions

* Python 3.6+
* Pytorch 1.6 +
* librosa
* soundfile
* pathlib
* Download the code:
```
git clone 
```

# Data

In the parameters to the main(see Params) you should specify three folders: Train,Valid,Test
The files inside were created by the "Libri speech files parser" , you can see here(https://github.com/Jenny-Smolenksy/libri-speech-files-parser)

Exmple for the data is provided in the data folder, and looks as follow:
```
+---test
|   +---AA
|   +---AE
|   +---AH
|   +---AO
|   +---AW
|   +---AY
|   +---EH
|   +---empty
|   +---ER
|   +---EY
|   +---IH
|   +---IY
|   +---OW
|   +---OY
|   +---UH
|   \---UW
+---train
|   +---AA
|   +---AE
|   +---AH
|   +---AO
|   +---AW
|   +---AY
|   +---EH
|   +---empty
|   +---ER
|   +---EY
|   +---IH
|   +---IY
|   +---OW
|   +---OY
|   +---UH
|   \---UW
\---valid
    +---AA
    +---AE
    +---AH
    +---AO
    +---AW
    +---AY
    +---EH
    +---empty
    +---ER
    +---EY
    +---IH
    +---IY
    +---OW
    +---OY
    +---UH
    \---UW

```

And inside each Vowel Folders(in the format mentioned [here](https://github.com/Jenny-Smolenksy/LibriSpeechFilesParser.git):
```
+---AA
|       10.wav
|       10.wrd
|       101.wav
|       101.wrd
|       102.wav
|       102.wrd
+---AE
|       100.wav
|       100.wrd
|       102.wav
|       102.wrd
|       103.wav
|       103.wrd
```

## Params

In the main function inside run_speech_yolo_vowels:
data_folder_train : with the training files (format as mentioned before, you can see example here)
data_folder_valid : with the valiation files (format as mentioned before, you can see example here)

SpeechYoloVowels get the folders and number of workrs
and create the model by the arguments

then run the train speech net to train the model
and in the end - run the test speech net to test the model accuracy on the test data set


# Output


In the output printed on the validation and test set, you can see the following output printed:
The first section:
The model predict for each signal and each cell if there is an object there
by taking all the cells that the model "correct object" (There is an object(vowel) there and also the model predicted there is vowel there)
it split all the "correct object" to classes(vowels):
For exmple, the printed line:
Class AA : correct : 8.0/9.0 (89%)
That's mean :

8 -> how many cells classes are AA **and**
    the model predict that there is object there **and**
	the model classify there is "AA" there
	
9 -> how many cells classes are AA **and**
	the model predict that there is object there

In the second section you can see how many mistakes was in the model prediction of "There is an object in the cell" (localization)
and after that the classification 


## Authors

 **Jenny Smolensky**  , **Almog Gueta** 
 
 
