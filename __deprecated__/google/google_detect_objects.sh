mkdir -p output/objects/

for var in "$@"
do
    echo "Processing $var"
    result="$(basename $var)"
    gcloud ml vision detect-objects $var > "output/objects/$result.json"
done
