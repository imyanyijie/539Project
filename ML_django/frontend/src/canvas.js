import React, { Component } from 'react';
import { v4 } from 'uuid';

class Canvas extends Component {
  constructor(props) {
    super(props);

    this.onMouseDown = this.onMouseDown.bind(this);
    this.onMouseMove = this.onMouseMove.bind(this);
    this.endPaintEvent = this.endPaintEvent.bind(this); 
    this.onSubmit = this.onSubmit.bind(this);

    this.state = {
            ans: "..."
    }
    this.sendPaintData = this.sendPaintData.bind(this); //************ add this
  }
  isPainting = false;
  userStrokeStyle = '#EE92C2';
  guestStrokeStyle = '#F0C987';
  line = [];
  userId = v4();
  prevPos = { offsetX: 0, offsetY: 0 };

  onMouseDown({ nativeEvent }) {
    const { offsetX, offsetY } = nativeEvent;
    this.isPainting = true;
    this.prevPos = { offsetX, offsetY };
  }

  onMouseMove({ nativeEvent }) {
    if (this.isPainting) {
      const { offsetX, offsetY } = nativeEvent;
      const offSetData = { offsetX, offsetY };
      this.position = {
        start: { ...this.prevPos },
        stop: { ...offSetData },
      };
      this.line = this.line.concat(this.position);

      this.paint(this.prevPos, offSetData, this.userStrokeStyle);
    }
  }

  endPaintEvent() {
    if (this.isPainting) {
      this.isPainting = false;
    //   this.sendPaintData();
    }

    // createImageBitmap(this.canvas).then(imageBitmap =>
    //     console.log(this.ctx.getImageData(0, 0, 256, 256))
    //     // this.ctx.drawImage(imageBitmap, 50, 50)
    // )


  }

  paint(prevPos, currPos, strokeStyle) {
    const { offsetX, offsetY } = currPos;
    const { offsetX: x, offsetY: y } = prevPos;

    this.ctx.beginPath();
    this.ctx.strokeStyle = strokeStyle;
    this.ctx.moveTo(x, y);
    this.ctx.lineTo(offsetX, offsetY);
    this.ctx.stroke();
    this.prevPos = { offsetX, offsetY };
  }
  onSubmit() {
    const canvas = this.canvas;
    // var image = canvas.toDataURL()
        // .replace("image/png", "image/octet-stream");
    this.sendPaintData()

  }
  // async
  sendPaintData= () => {

    let self = this
    //----------------------------------

    this.canvas.toBlob(function(blob) {
        const formData = new FormData();
        const image_id = Math.floor(Math.random() * (10000000000 - 0 + 1));
        const fn = image_id.toString() + '.png'
        formData.append('my-file', blob,  fn);
        // Post via axios or other transport method

        fetch('http://127.0.0.1:8000/api/draw_app/imagestore/create_image/', {
            // 'https://spartan-card-260220.appspot.com/api/draw_app/imagestore/create_image/'
            method: "POST",
            headers: {
                Accept: "application/json",
            },
            body: formData
        }).then((response) => { response.json().then((data) => { self.setState({ans: data.result}); });
        })
    });
    //--------------------------------------------------
    this.line = [];
  }

  componentDidMount() {
    this.canvas.width = 256;
    this.canvas.height = 256;
    this.ctx = this.canvas.getContext('2d');
    this.ctx.lineJoin = 'round';
    this.ctx.lineCap = 'round';
    this.ctx.lineWidth = 3;
  }

  render() {
    return (
      <div style={{flexDirection:"row"}}>
        <canvas
            ref={(ref) => (this.canvas = ref)}
            style={{ background: 'black' }}
            onMouseDown={this.onMouseDown}
            onMouseLeave={this.endPaintEvent}
            onMouseUp={this.endPaintEvent}
            onMouseMove={this.onMouseMove}
        />
        <button style={{marginLeft:"row"}}
            // ref={(ref) => (this.canvas = ref)}
            onClick={this.onSubmit}
            >Submit
        </button>

        <text>My guess is {this.state.ans}</text>
      </div>
    );
  }
}

export default Canvas;