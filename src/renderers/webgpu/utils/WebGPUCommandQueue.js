const _queues = new WeakMap();

const _wrappedMethods = [ 'submit', 'onSubmittedWorkDone', 'writeBuffer', 'writeTexture', 'copyExternalImageToTexture', 'copyElementImageToTexture' ];

/**
 * Defers command buffer submission so a frame reaches the GPU in as few
 * `queue.submit()` calls as possible. Pending command buffers are flushed at
 * the end of the frame, before a queue write would overwrite a resource that a
 * pending command buffer may still read, and before any readback.
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
		 * The number of command buffers appended so far. Used to stamp writes.
		 *
		 * @type {number}
		 */
		this.appended = 0;

		/**
		 * Maps each resource written during the current frame to the value of
		 * `appended` at its last write.
		 *
		 * @type {Map<GPUBuffer|GPUTexture,number>}
		 */
		this.writes = new Map();

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

		this.pending.push( commandBuffer );
		this.appended ++;

		this._queueFlush();

	}

	/**
	 * Records a queue write to the given resource. If the resource was already
	 * written this frame and a command buffer appended since then is still
	 * pending, the pending list is flushed first.
	 *
	 * @param {GPUBuffer|GPUTexture} resource - The written resource.
	 */
	write( resource ) {

		const stamp = this.writes.get( resource );

		if ( stamp !== undefined && stamp < this.appended && this.pending.length > 0 ) {

			this.flush();

		}

		this.writes.set( resource, this.appended );

		this._queueFlush();

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
	 * @param {boolean} [frameEnd=false] - Whether this flush ends the frame, which resets write tracking.
	 */
	flush( frameEnd = false ) {

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

		const destroyQueue = this.destroyQueue;

		if ( destroyQueue.length > 0 ) {

			for ( let i = 0; i < destroyQueue.length; i ++ ) destroyQueue[ i ].destroy();

			destroyQueue.length = 0;

		}

		if ( frameEnd === true ) this.writes.clear();

	}

	/**
	 * Drops all pending work after a device loss.
	 */
	lose() {

		this.isLost = true;
		this.pending.length = 0;
		this.writes.clear();

		for ( const resource of this.destroyQueue ) resource.destroy();

		this.destroyQueue.length = 0;

	}

	/**
	 * Flushes pending work and restores the original queue methods.
	 */
	dispose() {

		this.flush( true );

		const queue = this.queue;

		for ( const name of _wrappedMethods ) {

			if ( Object.prototype.hasOwnProperty.call( queue, name ) ) delete queue[ name ];

		}

		_queues.delete( this.device );

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

		queue.writeBuffer = function ( buffer ) {

			scope.write( buffer );

			return proto.writeBuffer.apply( queue, arguments );

		};

		queue.writeTexture = function ( destination ) {

			scope.write( destination.texture );

			return proto.writeTexture.apply( queue, arguments );

		};

		queue.copyExternalImageToTexture = function ( source, destination ) {

			scope.write( destination.texture );

			return proto.copyExternalImageToTexture.apply( queue, arguments );

		};

		if ( typeof proto.copyElementImageToTexture === 'function' ) {

			const copyElementImageToTexture = function () {

				const destination = arguments.length === 2 ? arguments[ 1 ].destination : arguments[ 3 ];

				if ( destination && destination.texture ) scope.write( destination.texture );

				return proto.copyElementImageToTexture.apply( queue, arguments );

			};

			Object.defineProperty( copyElementImageToTexture, 'length', { value: proto.copyElementImageToTexture.length } );

			queue.copyElementImageToTexture = copyElementImageToTexture;

		}

	}

}

export default WebGPUCommandQueue;
