const _queues = new WeakMap();

const _wrappedMethods = [ 'submit', 'onSubmittedWorkDone', 'writeBuffer', 'writeTexture', 'copyExternalImageToTexture', 'copyElementImageToTexture' ];

const _stagingMinSize = 262144;
const _stagingMaxSize = 4194304;

const _stagingDescriptor = { label: 'WebGPUCommandQueue.staging', size: 0, usage: 0 };
const _uploadEncoderDescriptor = { label: 'WebGPUCommandQueue.writes' };

/**
 * Returns the byte length a `writeBuffer()` call would write, or `-1` if the
 * arguments are not a valid in-range write.
 *
 * @private
 * @param {BufferSource} data - The source data.
 * @param {number} [dataOffset=0] - Offset into `data`, in elements for typed arrays and in bytes otherwise.
 * @param {number} [size] - Size of the write, in the same units as `dataOffset`.
 * @return {number} The byte length.
 */
function getWriteByteLength( data, dataOffset = 0, size = undefined ) {

	if ( data === null || typeof data !== 'object' ) return - 1;

	const elementSize = data.BYTES_PER_ELEMENT !== undefined ? data.BYTES_PER_ELEMENT : 1;
	const length = data.BYTES_PER_ELEMENT !== undefined ? data.length : data.byteLength;
	const count = size === undefined ? length - dataOffset : size;

	if ( Number.isInteger( length ) === false || Number.isInteger( dataOffset ) === false || Number.isInteger( count ) === false ) return - 1;
	if ( dataOffset < 0 || count < 0 || dataOffset + count > length ) return - 1;

	return count * elementSize;

}

/**
 * Defers command buffer submission so a frame reaches the GPU in as few
 * `queue.submit()` calls as possible. Pending command buffers are flushed at
 * the end of the frame and before any readback.
 *
 * A queue write executes before every command buffer submitted after it, so a
 * buffer write made while command buffers are pending is staged instead and
 * copied by a command buffer appended at that point, keeping the order of
 * immediate submission. Texture writes flush first.
 *
 * @private
 */
class WebGPUCommandQueue {

	/**
	 * Constructs a new command queue for the given device and wraps the
	 * device's `GPUQueue` so writes and external submits are ordered correctly.
	 *
	 * @param {GPUDevice} device - The GPU device.
	 * @param {?Function} [onError=null] - Receives validation errors raised by deferred submits, plus an optional explanatory message.
	 */
	constructor( device, onError = null ) {

		/**
		 * The GPU device.
		 *
		 * @type {GPUDevice}
		 */
		this.device = device;

		/**
		 * The wrapped GPU queue.
		 *
		 * @type {GPUQueue}
		 */
		this.queue = device.queue;

		/**
		 * Receives validation errors raised by deferred submits, plus an optional
		 * explanatory message.
		 *
		 * @type {?Function}
		 */
		this.onError = onError;

		/**
		 * Command buffers waiting to be submitted.
		 *
		 * @type {Array<GPUCommandBuffer>}
		 */
		this.pending = [];

		/**
		 * Records the copies of staged buffer writes made since the last pending
		 * command buffer. It is finished and appended before the next one.
		 *
		 * @type {?GPUCommandEncoder}
		 */
		this.uploadEncoder = null;

		/**
		 * Holds the data of staged buffer writes until the next flush.
		 *
		 * @type {?GPUBuffer}
		 */
		this.staging = null;

		/**
		 * Bytes of `staging` used since the last flush.
		 *
		 * @type {number}
		 */
		this.stagingOffset = 0;

		/**
		 * Resources whose destruction waits for the next flush.
		 *
		 * @type {Array<GPUBuffer|GPUTexture|GPUQuerySet>}
		 */
		this.destroyQueue = [];

		/**
		 * Set when the device is lost; pending work is dropped from then on.
		 *
		 * @type {boolean}
		 */
		this.isLost = false;

		/**
		 * Remaining frame-end flushes with work that submit each command buffer on its own.
		 * Set after a batched submit fails, since one invalid command buffer makes
		 * `queue.submit()` reject the whole batch.
		 *
		 * @type {number}
		 */
		this.fallbackFlushes = 0;

		/**
		 * Length of the next per-command-buffer fallback window, in frame-end flushes. Doubles
		 * each time batching is retried and fails again.
		 *
		 * @type {number}
		 */
		this.fallbackLength = 120;

		this._flushQueued = false;
		this._fallbackLogged = false;
		this._fallbackWindow = 0;
		this._single = [ null ];

		this._frameFlush = () => {

			this._flushQueued = false;
			this.flush( true );

		};

		this._onSubmitError = ( err ) => {

			if ( err === null ) return;

			if ( this.fallbackFlushes > 0 ) this.fallbackFlushes = this._fallbackWindow;

			if ( this.onError !== null ) this.onError( err );

		};

		this._onBatchSubmitError = ( err ) => {

			if ( err === null || this.fallbackFlushes > 0 ) return;

			this._fallbackWindow = this.fallbackLength;
			this.fallbackFlushes = this._fallbackWindow;
			this.fallbackLength = Math.min( this.fallbackLength * 2, 7680 );

			if ( this.onError === null ) return;

			if ( this._fallbackLogged === false ) {

				this._fallbackLogged = true;

				this.onError( err, `${ err.message } (A batched queue.submit() was rejected because one of its command buffers is invalid, so all work in that batch was lost. Command buffers are now submitted individually for ${ this.fallbackFlushes } frames before batching is retried.)` );

			} else {

				this.onError( err );

			}

		};

		this._install();

		_queues.set( device, this );

	}

	/**
	 * Returns the command queue registered for the given device.
	 *
	 * @param {GPUDevice} device - The GPU device.
	 * @return {?WebGPUCommandQueue} The command queue, or `null`.
	 */
	static get( device ) {

		return _queues.get( device ) || null;

	}

	/**
	 * Appends a command buffer to the pending list.
	 *
	 * @param {GPUCommandBuffer} commandBuffer - The command buffer.
	 */
	submit( commandBuffer ) {

		if ( this.isLost === true ) return;

		this._closeUploads();

		this.pending.push( commandBuffer );

		this._queueFlush();

	}

	/**
	 * Stages a buffer write while command buffers are pending, so it executes
	 * after them and before command buffers appended later. Returns `false` when
	 * the caller must write directly, after flushing if ordering requires it.
	 *
	 * @param {GPUBuffer} buffer - The destination buffer.
	 * @param {number} bufferOffset - The byte offset into `buffer`.
	 * @param {BufferSource} data - The source data.
	 * @param {number} [dataOffset] - Offset into `data`, in elements for typed arrays and in bytes otherwise.
	 * @param {number} [size] - Size of the write, in the same units as `dataOffset`.
	 * @return {boolean} Whether the write was staged.
	 */
	stageWrite( buffer, bufferOffset, data, dataOffset, size ) {

		if ( this.pending.length === 0 || this.isLost === true ) return false;

		const byteLength = getWriteByteLength( data, dataOffset, size );

		if ( byteLength === 0 ) return false;

		if ( byteLength < 0 || byteLength > _stagingMaxSize || Number.isInteger( bufferOffset ) === false || bufferOffset < 0 || ( bufferOffset % 4 ) !== 0 || ( byteLength % 4 ) !== 0 ||
			bufferOffset + byteLength > buffer.size || ( buffer.usage & GPUBufferUsage.COPY_DST ) === 0 || buffer.mapState !== 'unmapped' ) {

			this.flush();

			return false;

		}

		let staging = this.staging;

		if ( staging === null || this.stagingOffset + byteLength > staging.size ) {

			if ( staging !== null && staging.size >= _stagingMaxSize ) {

				this.flush();

				return false;

			}

			let stagingSize = staging === null ? _stagingMinSize : staging.size * 2;

			while ( stagingSize < byteLength ) stagingSize *= 2;

			if ( staging !== null ) this.destroyQueue.push( staging );

			_stagingDescriptor.size = Math.min( stagingSize, _stagingMaxSize );
			_stagingDescriptor.usage = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

			staging = this.device.createBuffer( _stagingDescriptor );

			this.staging = staging;
			this.stagingOffset = 0;

		}

		const offset = this.stagingOffset;

		Object.getPrototypeOf( this.queue ).writeBuffer.call( this.queue, staging, offset, data, dataOffset, size );

		if ( this.uploadEncoder === null ) this.uploadEncoder = this.device.createCommandEncoder( _uploadEncoderDescriptor );

		this.uploadEncoder.copyBufferToBuffer( staging, offset, buffer, bufferOffset, byteLength );

		this.stagingOffset = offset + byteLength;

		this._queueFlush();

		return true;

	}

	/**
	 * Flushes pending command buffers before a queue write to a texture, which
	 * would otherwise execute ahead of them.
	 */
	writeTexture() {

		if ( this.pending.length > 0 ) this.flush();

	}

	/**
	 * Destroys the given resource, after the next flush if command buffers
	 * that may reference it are still pending.
	 *
	 * @param {GPUBuffer|GPUTexture|GPUQuerySet} resource - The resource.
	 */
	destroy( resource ) {

		if ( this.pending.length > 0 ) {

			this.destroyQueue.push( resource );

		} else {

			resource.destroy();

		}

	}

	/**
	 * Submits all pending command buffers in one `queue.submit()` call.
	 *
	 * @param {boolean} [frameEnd=false] - Whether this flush ends the frame, which counts down the per-command-buffer fallback.
	 */
	flush( frameEnd = false ) {

		this._closeUploads();

		const pending = this.pending;

		if ( pending.length > 0 && this.isLost === false ) {

			const queue = this.queue;
			const device = this.device;

			const submitGPU = Object.getPrototypeOf( queue ).submit;

			device.pushErrorScope( 'validation' );

			if ( this.fallbackFlushes > 0 || pending.length === 1 ) {

				const single = this._single;

				for ( let i = 0; i < pending.length; i ++ ) {

					single[ 0 ] = pending[ i ];
					submitGPU.call( queue, single );

				}

				single[ 0 ] = null;

				device.popErrorScope().then( this._onSubmitError );

				if ( frameEnd === true && this.fallbackFlushes > 0 ) this.fallbackFlushes --;

			} else {

				submitGPU.call( queue, pending );

				device.popErrorScope().then( this._onBatchSubmitError );

			}

		}

		pending.length = 0;
		this.stagingOffset = 0;

		const destroyQueue = this.destroyQueue;

		if ( destroyQueue.length > 0 ) {

			for ( let i = 0; i < destroyQueue.length; i ++ ) destroyQueue[ i ].destroy();

			destroyQueue.length = 0;

		}

	}

	/**
	 * Drops all pending work after a device loss.
	 */
	lose() {

		this.isLost = true;
		this.pending.length = 0;
		this.uploadEncoder = null;
		this.stagingOffset = 0;

		for ( const resource of this.destroyQueue ) resource.destroy();

		this.destroyQueue.length = 0;

		if ( this.staging !== null ) {

			this.staging.destroy();
			this.staging = null;

		}

	}

	/**
	 * Flushes pending work and restores the original queue methods.
	 */
	dispose() {

		this.flush( true );

		if ( this.staging !== null ) {

			this.staging.destroy();
			this.staging = null;

		}

		const queue = this.queue;

		for ( const name of _wrappedMethods ) {

			if ( Object.prototype.hasOwnProperty.call( queue, name ) ) delete queue[ name ];

		}

		_queues.delete( this.device );

	}

	/**
	 * Appends the command buffer holding the copies of staged writes, if any.
	 *
	 * @private
	 */
	_closeUploads() {

		if ( this.uploadEncoder !== null ) {

			const uploadEncoder = this.uploadEncoder;

			this.uploadEncoder = null;

			if ( this.isLost === false ) this.pending.push( uploadEncoder.finish() );

		}

	}

	/**
	 * Schedules a frame-end flush at the next microtask checkpoint, which runs
	 * before the current task ends and the canvas texture is presented.
	 *
	 * @private
	 */
	_queueFlush() {

		if ( this._flushQueued === false ) {

			this._flushQueued = true;

			queueMicrotask( this._frameFlush );

		}

	}

	/**
	 * Wraps the queue methods as own properties of the queue instance. The
	 * prototype methods are looked up at call time so later wrappers still apply.
	 *
	 * @private
	 */
	_install() {

		const scope = this;
		const queue = this.queue;
		const proto = Object.getPrototypeOf( queue );

		queue.submit = function ( commandBuffers ) {

			scope.flush();

			return proto.submit.call( queue, commandBuffers );

		};

		queue.onSubmittedWorkDone = function () {

			scope.flush();

			return proto.onSubmittedWorkDone.call( queue );

		};

		queue.writeBuffer = function ( buffer, bufferOffset, data, dataOffset, size ) {

			if ( scope.stageWrite( buffer, bufferOffset, data, dataOffset, size ) === true ) return;

			return proto.writeBuffer.apply( queue, arguments );

		};

		queue.writeTexture = function () {

			scope.writeTexture();

			return proto.writeTexture.apply( queue, arguments );

		};

		queue.copyExternalImageToTexture = function () {

			scope.writeTexture();

			return proto.copyExternalImageToTexture.apply( queue, arguments );

		};

		if ( typeof proto.copyElementImageToTexture === 'function' ) {

			const copyElementImageToTexture = function () {

				scope.writeTexture();

				return proto.copyElementImageToTexture.apply( queue, arguments );

			};

			Object.defineProperty( copyElementImageToTexture, 'length', { value: proto.copyElementImageToTexture.length } );

			queue.copyElementImageToTexture = copyElementImageToTexture;

		}

	}

}

export default WebGPUCommandQueue;
