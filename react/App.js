/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 * @flow
 */

import React, { Component } from 'react';
import {
    AppRegistry,
    Dimensions,
    StyleSheet,
    Text,
    TouchableHighlight,
    View,
    Platform,
    Image
  } from 'react-native';
import Camera from 'react-native-camera';
import RNNode from 'react-native-node';

export default class BadInstagramCloneApp extends Component {

  constructor() {
    super();
    this.state = { msg: "node server broke", isLastPic: false, lastPic: null};
  }

  componentDidMount() {
    console.log("asdf");
    RNNode.start(['--port', '5001']);
  }

  sendRequest() {
    const data = new FormData();
    data.append('photo', {
      uri: 'file:///storage/emulated/0/DCIM/IMG_20180120_110956.jpg',
      type : 'image/jpg',
      name : "gaurav.jpg"
    });
    fetch("http://gauravity.com/node", {
      method: "POST",
      body: data
    }).then(res => console.log(res))
      .catch(err => console.log(err));
  }

  render() {

    let pic = {
      uri: 'file:///storage/emulated/0/DCIM/IMG_20180120_110956.jpg'
    };


    return (
      <View style={styles.container}>
        <Camera
          ref={(cam) => {
            this.camera = cam;
          }}
          style={styles.preview}
          aspect={Camera.constants.Aspect.fill}
          playSoundOnCapture={false}
          >
          <Text style={styles.capture} onPress={this.takePicture.bind(this)}>[CAPTURE]</Text>
        </Camera>
        <Image source = {pic} style = {{width: 100, height: 100}}/>
        <Text style={styles.button} onPress={() => this.sendRequest()}>
          {this.state.msg}
        </Text>
      </View>
    );
  }

  takePicture() {
    const options = {};
    //options.location = ...
    console.log("picture taken");
    this.camera.capture({metadata: options})
      .then((data) => console.log(data))
      .catch(err => console.error(err));
  }

}

AppRegistry.registerComponent('BadInstagramCloneApp', () => BadInstagramCloneApp);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5FCFF',
  },
  welcome: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
  instructions: {
    textAlign: 'center',
    color: '#333333',
    marginBottom: 5,
  },
  preview: {
    flex: 1,
    justifyContent: 'flex-end',
    alignItems: 'center',
    width: 400
  },
  capture: {
    flex: 0,
    backgroundColor: '#fff',
    borderRadius: 5,
    color: '#000',
    padding: 10,
    margin: 40
  },
  button: {
  color: "white",
  fontSize: 20,
  marginTop: 40,
  backgroundColor: "#5c7cfa",
  borderRadius: 4,
  padding: 8,
  textAlign: "center"
  }
});
