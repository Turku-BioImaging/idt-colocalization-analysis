
dirRaw = getDirectory("Choose a Directory ALL");
listRaw = getFileList(dirRaw);
n1 = lengthOf(listRaw);

dirSave = getDirectory("Choose a Directory SAVE");

//---------------------------------


crop_n_mask();


//---------------------------------



function crop_n_mask(){

    setBatchMode(true);
    for (i=0; i<n1; i++) {

	pathRaw = dirRaw+listRaw[i];
	open(pathRaw);

	//split channels and close C2
	selectedFile = getTitle();
	run("Split Channels");
	channel1 = "C1"+"-"+selectedFile;
	channel2 = "C2"+"-"+selectedFile;
	selectImage(channel2);	
	close;

//pri-processing
	selectImage(channel1);
	
	run("Kuwahara Filter", "sampling=7 stack");
	run("Variance...", "radius=10 stack");

	setAutoThreshold("Li dark stack");
	run("Convert to Mask", "method=Li background=Dark black");
	

//pixel operations

run("Fill Holes", "stack");
run("Analyze Particles...", "size=100-Infinity show=Masks stack");
run("Invert LUT");

run("Options...", "iterations=4 count=1 black do=Dilate stack");

run("Fill Holes", "stack");

close("\\Others");


	selectedFile = replace(selectedFile, ".tif", "");
	saveAs("Tiff", dirSave + selectedFile + "_mask");

	close("*");

	setBatchMode("False");
	}
}

