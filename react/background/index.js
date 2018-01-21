const express = require("express");

const app = express();

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
console.log("node");

app.post("/", (req, res) => {
  res.send(req.body);
}).catch(err => console.log(err));

app.listen(5001);
