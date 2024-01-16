# Unzip all tcx.gz s
for file in ./Raw_Strava_Data/activities/*.tcx.gz; do
    gzip -d "$file"
done
# Move .tcx s
for file in ./Raw_Strava_Data/activities/*.tcx; do
    mv "$file" Raw_Strava_Data/uzip_activities
done
#delete the first 10 chrs
for file in ./Raw_Strava_Data/uzip_activities/*.tcx; do
    a=$(cat "$file")
    echo ${a:10} > "$file"
done