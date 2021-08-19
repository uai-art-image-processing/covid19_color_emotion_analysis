mkdir -p output/properties/

for var in "$@"
do
    echo "Processing $var"
    result="$(basename $var)"
    gcloud ml vision detect-image-properties $var > "output/properties/$result.json"
done
