var http = require('http');
const googleTrends = require('google-trends-api');
var fs = require('fs');

http.createServer(function (req, res) {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.end('Hello World!');
}).listen(8080);

googleTrends.interestOverTime({keyword: 'ethereum'})
.then(function(results) {
  console.log(results);
})
.catch(function(err) {
  console.error(err);
});

fs.writeFile("tmp.txt", "Hey there!", function(err) {
    if(err) {
        return console.log(err);
    }

    console.log("The file was saved!");
}); 