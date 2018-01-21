const cv = require('opencv');
const UPPER_CASCADE="haarcascadetorso.xml"

cv.readImage('../otsukevin2.png', function(err, img) {
  if(err) throw err;
  if(img.width() < 1 || img.height() < 1) {
    throw new Error('Image has no size');
  }

  img.detectObject(UPPER_CASCADE, {}, function(err, faces) {
    if (err) throw err;
    //for (var i = 0; i < faces.length; i++)
      var face = faces[0];
      img.ellipse(face.x + face.width / 2, face.y + face.height / 2, face.width / 2, face.height / 2);
    img.save('changed.png');
  });
});

console.log("Hello Wolrd");
