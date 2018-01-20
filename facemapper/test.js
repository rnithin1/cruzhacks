const cv = require('opencv');

cv.readImage('./img/myImage.jpg', function (err, img) {
  if (err) {
    throw err;
  }

  const width = im.width();
  const height = im.height();

  if (width < 1 || height < 1) {
    throw new Error('Image has no size');
  }

  // do some cool stuff with img

  // save img
  img.save('./img/myNewImage.jpg');
});
