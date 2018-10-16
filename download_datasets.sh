FILE=$1

if [[ $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" &&  $FILE != "photo2art" && $FILE != "edges2shoes" && $FILE != "edges2handbags" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "vangogh2photo" ]]; then
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, photo2art, edge2shoes"
    exit 1
fi

if [[ $FILE == "photo2art" ]]; then
    mkdir -p ./datasets/$FILE
    Files=(monet2photo cezanne2photo vangogh2photo)
    Domain=(B C D)
    for ((i = 0; i <= 2; i++))
    do
        URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/${Files[$i]}.zip
        ZIP_FILE=./datasets/${Files[$i]}.zip
        wget -N $URL -O $ZIP_FILE
        mkdir -p ./datasets/${Files[$i]}
        unzip $ZIP_FILE -d ./datasets/
        if [[ ${Files[$i]} == "monet2photo" ]]; then
            mv ./datasets/monet2photo/trainB  ./datasets/photo2art/trainA
            mv ./datasets/monet2photo/testB  ./datasets/photo2art/testA
        fi
        mv ./datasets/${Files[$i]}/trainA  ./datasets/photo2art/train${Domain[$i]}
        mv ./datasets/${Files[$i]}/testA  ./datasets/photo2art/test${Domain[$i]}
        rm -rf ./datasets/${Files[$i]}
        rm $ZIP_FILE
    done

elif [[ $FILE == "edges2shoes" ||  $FILE == "edges2handbags" ]]; then
    URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
    TAR_FILE=./datasets/$FILE.tar.gz
    wget -N $URL -O $TAR_FILE
    mkdir -p ./datasets/$FILE
    tar -zxvf $TAR_FILE -C ./datasets/
    echo "Start preprocessing dataset..."
    python ./split.py ./datasets/$FILE
    echo "Finished preprocessing dataset."
    if [ -d "./datasets/$FILE/testB" ];then
        rm -rf ./datasets/$FILE/train
        rm -rf ./datasets/$FILE/val
        rm $TAR_FILE
    fi
else
    URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
    ZIP_FILE=./datasets/$FILE.zip
    wget -N $URL -O $ZIP_FILE
    mkdir -p ./datasets/$FILE
    unzip $ZIP_FILE -d ./datasets/
    rm $ZIP_FILE
fi