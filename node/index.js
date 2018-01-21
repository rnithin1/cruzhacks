const express = require("express");
const bodyParser = require('body-parser');
const multer = require('multer');
const storage = multer.diskStorage({
  destination: './uploads',
  filename: function (req, file, callback) {
    return "asdf.jpg";
  }
});


const app = express();

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
console.log("node");

app.post("/",upload.single('photo'), (req, res) => {
  res.send(req.body);
});

app.listen(5001);
