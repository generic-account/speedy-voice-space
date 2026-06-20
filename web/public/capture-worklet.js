// AudioWorklet capture processor.
// Coalesces the 128-sample render quantum into fixed-size mono blocks and posts
// each block (zero-copy transfer) to the main thread / engine.
//
// Plain JS (not bundled) so it loads reliably via audioWorklet.addModule().

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.blockSize = opts.blockSize || 1024;
    this._buf = new Float32Array(this.blockSize);
    this._fill = 0;
  }

  process(inputs) {
    const input = inputs[0];
    // input[0] is the first channel; absent if the source is disconnected.
    if (input && input[0]) {
      const ch = input[0];
      for (let i = 0; i < ch.length; i++) {
        this._buf[this._fill++] = ch[i];
        if (this._fill >= this.blockSize) {
          const out = this._buf.slice(0); // copy out, keep _buf for reuse
          // Tag the block with its capture frame index from the audio clock
          // (currentFrame is the context's running sample count). This is the
          // true real-time position, independent of any downstream worker
          // backlog — so the display clock never lurches when the backlog drains.
          const frame = currentFrame + i;
          this.port.postMessage({ type: "block", samples: out, frame }, [out.buffer]);
          this._fill = 0;
        }
      }
    }
    // Returning true keeps the node alive. We write nothing to outputs, so the
    // node is silent even though it's connected to destination (no feedback).
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
