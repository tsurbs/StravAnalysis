# for file in ../activities/*.tcx.gz; do
#     gzip -d "$file"
# done
# for file in ../activities/*.tcx; do
#     mv "$file" ./The_Data
# done
for file in ./The_Data/*.tcx; do
    a=$(cat "$file")
    echo ${a:10} > "$file"
done