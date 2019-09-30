rm -rf train
mkdir train && cd train
END=80
for ((i=0; i<END; i++)) do
    code="000"
    if [ $i -lt 10 ]
    then
        code=$code"0"$i
    else
        code=$code$i
    fi
    wget https://storage.googleapis.com/brain-genomics-public/research/proteins/pfam/random_split/train/data-$code-of-000$END
done
cd ..

rm -rf test
mkdir test && cd test
END=10
for ((j=0; j<END; j++)) do
    code="000"
    if [ $j -lt 10 ]
    then
        code=$code"0"$j
    else
        code=$code$j
    fi
    wget https://storage.googleapis.com/brain-genomics-public/research/proteins/pfam/random_split/test/data-$code-of-000$END
done
cd ..

rm -rf dev
mkdir dev && cd dev
END=10
for ((k=0; k<END; k++)) do
    code="000"
    if [ $k -lt 10 ]
    then
        code=$code"0"$k
    else
        code=$code$k
    fi
    wget https://storage.googleapis.com/brain-genomics-public/research/proteins/pfam/random_split/dev/data-$code-of-000$END
done
