ACCESS_KEY="-qXfrx1DeYQ8UOpa4pVbY7XdKcPqK7xsoVng6APin08"

curl -s "https://api.unsplash.com/photos/random?count=2&query=rainbow&orientation=landscape" \
  -H "Authorization: Client-ID $ACCESS_KEY" \
  | jq -r '.[].urls.raw' | while read url; do
    wget -q "${url}&fm=png&w=7680&h=4320" -O "pistachio_$(date +%s%N).png"
done
