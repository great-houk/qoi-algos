ACCESS_KEY="-qXfrx1DeYQ8UOpa4pVbY7XdKcPqK7xsoVng6APin08"

curl -s "https://api.unsplash.com/photos/random?count=5&query=pistachios%20nut&orientation=landscape" \
  -H "Authorization: Client-ID $ACCESS_KEY" \
  | jq -r '.[].urls.raw' | while read url; do
    wget -q "${url}&fm=png&w=1920&h=1080" -O "pistachio_$(date +%s%N).png"
done
