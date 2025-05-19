import easygui
import os
import subprocess


def trainTransformer(sequenceLength=40, epochs=50, learningRate=0.001, timeInterval=30, dropout=0.1, headNum=8, encodingLayers=3, dModel=64, posEncoding='fixed', patience=3, dimensions=14):
    FSPattern = "-FS-" if dimensions != 14 else ""

    # Execute transformer in another directory
    directoryToTransformerSRC = "../mvts_transformer/mvts_transformer-master/src/"
    outputDirectory = "output/CloudWorkload/Sequence"+str(sequenceLength)+"/"
    dataDirectory = os.getcwd()

    os.chdir(directoryToTransformerSRC)

    # Create output directory
    if not os.path.isdir(os.getcwd()+"/"+outputDirectory):
        os.mkdir(os.getcwd()+"/"+outputDirectory)

    name = "TRAIN_Cloudwork_prediction+epoch="+str(epochs)+"+seqLen="+str(sequenceLength) + "+dimensions="+str(dimensions)\
        + "+LR="+str(learningRate)+"+timeInt="+str(timeInterval)+"+dropout="+str(dropout)\
        + "+noHead="+str(headNum)+"+numEnc="+str(encodingLayers) + \
        "+dModel="+str(dModel)+"+posEnc="+str(posEncoding) + \
        "+patience="+str(patience)

    # Build command
    command = "python main.py --output_dir "+outputDirectory+" --comment \"regression from Scratch\" --name "+name+" --records_file Regression_records.xls \
    --data_dir "+dataDirectory+" --data_class gtc --pattern TRAIN"+FSPattern+" --val_pattern VALIDATION"+FSPattern+" --test_pattern TEST"+FSPattern+" --epochs "+str(epochs)+" --lr "+str(learningRate)+" --optimizer RAdam \
     --pos_encoding learnable --max_seq_len "+str(sequenceLength)+" --task regression\
    --num_heads "+str(headNum)+" --num_layers "+str(encodingLayers)+" --dropout "+str(dropout)+" --d_model "+str(dModel)+" --pos_encoding "+str(posEncoding)+" --patience "+str(patience) + " --dimensions "+str(dimensions)

    # Execute transformer
    process = subprocess.Popen(command)
    process.communicate()
    process.terminate()
    os.chdir(dataDirectory)

# Can only be done after training


def testTransformer(sequenceLength=40, epochs=50, learningRate=0.001, timeInterval=30, dropout=0.1, headNum=8, encodingLayers=3, dModel=64, posEncoding='fixed', patience=3, dimensions=14):
    FSPattern = "-FS-" if dimensions != 14 else ""

    # Execute transformer in another directory
    directoryToTransformerSRC = "../mvts_transformer/mvts_transformer-master/src/"
    outputDirectory = "output/CloudWorkload/Sequence"+str(sequenceLength)+"/"
    dataDirectory = os.path.dirname(os.path.realpath(__file__))

    currentDIrectory = os.getcwd()
    if ("\mvts_transformer\mvts_transformer-master\src" not in currentDIrectory):
        os.chdir(directoryToTransformerSRC)

    modelDirectory = easygui.fileopenbox(
        default=outputDirectory, title="Look for 'path to checkout/model_best.pth' or similar")

    # Create output directory
    if not os.path.isdir(os.getcwd()+"/"+outputDirectory):
        os.mkdir(os.getcwd()+"/"+outputDirectory)

    name = "TEST_Cloudwork_prediction+epoch="+str(epochs)+"+seqLen="+str(sequenceLength) + "+dimensions="+str(dimensions)\
        + "+LR="+str(learningRate)+"+timeInt="+str(timeInterval)+"+dropout="+str(dropout)\
        + "+noHead="+str(headNum)+"+numEnc="+str(encodingLayers) + \
        "+dModel="+str(dModel)+"+posEnc="+str(posEncoding) + \
        "+patience="+str(patience)
    # Build command
    command = "python main.py --output_dir "+outputDirectory+" --comment \"regression from Scratch\" --name "+name+" --records_file Regression_records.xls \
    --data_dir "+dataDirectory+" --data_class gtc --pattern TRAIN"+FSPattern+" --test_pattern TEST"+FSPattern+" --test_only testset --load_model "+modelDirectory+" --optimizer RAdam \
    --pos_encoding learnable --max_seq_len "+str(sequenceLength)+" --task regression --dimensions "+str(dimensions)+" --val_ratio 0"

    # Execute transformer
    process = subprocess.Popen(command)
    process.communicate()
    process.terminate()


def trainSlidingWindowTransformer(sequenceLength=40, epochs=50, learningRate=0.001, timeInterval=30, dropout=0.1, headNum=8, encodingLayers=3, dModel=64, posEncoding='fixed', patience=3, dimensions=14):
    FSPattern = "-FS-" if dimensions != 14 else ""
    windowLength = 100
    windowStep = 10

    # Execute transformer in another directory
    directoryToTransformerSRC = "../transformerSlidingWindow/src/"
    outputDirectory = "output/CloudWorkload/Sequence"+str(sequenceLength)+"/"
    dataDirectory = os.getcwd()

    os.chdir(directoryToTransformerSRC)

    # Create output directory
    if not os.path.isdir(os.getcwd()+"/"+outputDirectory):
        os.mkdir(os.getcwd()+"/"+outputDirectory)

    name = "SW+seqLen="+str(sequenceLength) + "+dimensions="+str(dimensions)\
        + "+LR="+str(learningRate)+"+timeInt="+str(timeInterval)+"+dropout="+str(dropout)\
        + "+noHead="+str(headNum)+"+numEnc="+str(encodingLayers) + \
        "+dModel="+str(dModel)+"+posEnc="+str(posEncoding) + \
        "+patience="+str(patience)+"+windowLength=" + \
        str(windowLength)+"+windowStep="+str(windowStep)

    # Build command
    command = "python main.py --output_dir "+outputDirectory+" --comment \"regression from Scratch\" --name "+name+" --records_file Regression_records.xls \
    --data_dir "+dataDirectory+" --data_class gtc --pattern TRAIN"+FSPattern+" --test_pattern TEST"+FSPattern+" --val_pattern VAL"+FSPattern+" --epochs "+str(epochs)+" --lr "+str(learningRate)+" --optimizer RAdam \
     --pos_encoding learnable --max_seq_len "+str(sequenceLength)+" --task regression\
    --num_heads "+str(headNum)+" --num_layers "+str(encodingLayers)+" --dropout "+str(dropout)+" --d_model "+str(dModel)+" --pos_encoding "+str(posEncoding)+" --patience "+str(patience) + " --dimensions "+str(dimensions)

    # Execute transformer
    process = subprocess.Popen(command)
    process.communicate()
    process.terminate()
    os.chdir(dataDirectory)


def testSlidingWindowTransformer(sequenceLength=40, epochs=50, learningRate=0.001, timeInterval=30, dropout=0.1, headNum=8, encodingLayers=3, dModel=64, posEncoding='fixed', patience=3, dimensions=14):
    FSPattern = "-FS-" if dimensions != 14 else ""
    windowLength = 100
    windowStep = 10

    # Execute transformer in another directory
    directoryToTransformerSRC = "../transformerSlidingWindow/src/"
    outputDirectory = "output/CloudWorkload/Sequence"+str(sequenceLength)+"/"
    dataDirectory = os.path.dirname(os.path.realpath(__file__))

    os.chdir(directoryToTransformerSRC)

    modelDirectory = easygui.fileopenbox(
        default=outputDirectory, title="Look for 'path to checkout/model_best.pth' or similar")

    # Create output directory
    if not os.path.isdir(os.getcwd()+"/"+outputDirectory):
        os.mkdir(os.getcwd()+"/"+outputDirectory)

    name = "TEST_SW+seqLen="+str(sequenceLength) + "+dimensions="+str(dimensions)\
        + "+LR="+str(learningRate)+"+timeInt="+str(timeInterval)+"+dropout="+str(dropout)\
        + "+noHead="+str(headNum)+"+numEnc="+str(encodingLayers) + \
        "+dModel="+str(dModel)+"+posEnc="+str(posEncoding) + \
        "+patience="+str(patience)+"+windowLength=" + \
        str(windowLength)+"+windowStep="+str(windowStep)

    # Build command
    command = "python main.py --output_dir "+outputDirectory+" --comment \"regression from Scratch\" --name "+name+" --records_file Regression_records.xls \
    --data_dir "+dataDirectory+" --data_class gtc --pattern TRAIN"+FSPattern+" --test_pattern TEST"+FSPattern+" --epochs "+str(epochs)+" --lr "+str(learningRate)+" --optimizer RAdam \
     --pos_encoding learnable --max_seq_len "+str(sequenceLength)+" --test_only testset --load_model "+modelDirectory+" --task regression\
    --num_heads "+str(headNum)+" --num_layers "+str(encodingLayers)+" --dropout "+str(dropout)+" --d_model "+str(dModel)+" --pos_encoding "+str(posEncoding)+" --patience "+str(patience) + " --dimensions "+str(dimensions)

    # Execute transformer
    process = subprocess.Popen(command)
    process.communicate()
    process.terminate()
