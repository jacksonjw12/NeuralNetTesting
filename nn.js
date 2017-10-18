function Network(layout){
	this.learningRate = .5
	this.layers = []
	
	var prevLayer
	for(var l = 0; l < layout.length; l++){
		this.layers.push(new Layer(layout[l], prevLayer))
		prevLayer = this.layers[l]
	}

	this.setInputs = function(inputs){
		if(this.layers.length == 0){
			return;
		}
		this.layers[0].setOutputs(inputs)
	}
	this.forward = function(inputs){
		this.setInputs(inputs)
		for(var l = 1; l<this.layers.length; l++){
			this.layers[l].forward()//previous outputs are cleared before new ones are forwarded
		}

		return this.layers[this.layers.length-1].getOutputs()
	}
	this.backward = function(outputs){
		//find error of output layer
		for(var n = 0; n< outputs.length; n++){
			var outputNeuron = this.layers[this.layers.length-1].neurons[n]
			//error
			outputNeuron.error = this.errorFunc(outputNeuron.output,outputs[n])
			//gradient (derivative of sigmoid)
			var grad = outputNeuron.output * (1 - outputNeuron.output)
			//now multiply the error by the gradient
			outputNeuron.error *= grad
		}
		//find error in all other layers by taking weighted average of neurons in layer above(right)
		for(var l = this.layers.length-2; l > 0; l--){
			for(var n = 0; n < this.layers[l].neurons.length; n++){
				var hiddenNeuron = this.layers[l].neurons[n]
				hiddenNeuron.error = 0
				for(var nextLayerNeuron = 0; nextLayerNeuron < this.layers[l+1].neurons.length; nextLayerNeuron++){
					var nextNeuron = this.layers[l+1].neurons[nextLayerNeuron]
					//the error for this neuron is sum of errors of this next layers neurons in proportion to weight
					hiddenNeuron.error += nextNeuron.error * nextNeuron.weights[n].weight 
				}
				//gradient! similar to output layers
				var grad = hiddenNeuron.output * (1-hiddenNeuron.output)
				hiddenNeuron.error *= grad

			}
		}
		//Now that we have the error for each neuron we must change the weights and biases for them
		for(var l = this.layers.length-1; l > 0; l--){//include output layer in this loop
			for(var n = 0; n < this.layers[l].neurons.length; n++){
				var hiddenNeuron = this.layers[l].neurons[n]

				for(var prevLayerNeuron = 0; prevLayerNeuron < this.layers[l-1].neurons.length; prevLayerNeuron++){
					var prevNeuron = this.layers[l-1].neurons[prevLayerNeuron]
					//the error for this neuron is sum of errors of this next layers neurons in proportion to weight
					var deltaWeight = prevNeuron.output * hiddenNeuron.error
					hiddenNeuron.weights[prevLayerNeuron].delta += deltaWeight
					
				}
				//now time to update the neurons bias
				hiddenNeuron.bias.delta += hiddenNeuron.error * hiddenNeuron.bias.weight
				//this has to do with some weird calc where bias turns out to be just 1 * above

			}
		}

	}
	this.update = function(){
		for(var l = 0; l < this.layers.length; l++){
			this.layers[l].update(this.learningRate)
		}
	}
	this.errorFunc = function(val, target){
		return target - val
	}

	this.train = function(input, output, log){
		log = (typeof log !== 'undefined') ?  log : false;
		if(log){
			console.log(this.forward(input))
		}
		else{
			this.forward(input)
		}
		this.backward(output)
		this.update()
	}
	this.trainBatch = function(inputs, outputs, log, average){
		//same as above, but only call update at the end, option to average delta by the num of training data
	}
}

function Layer(length, prevLayer){
	this.neurons = []
	this.prevLayer = prevLayer
	for(var n = 0; n< length; n++){
		this.neurons.push( new Neuron(this.prevLayer))
	}

	this.setOutputs = function(outputs){
		if(outputs.length != this.neurons.length){
			console.err("wrong input length")
			return;
		}
		for(var n = 0; n< this.neurons.length; n++){

			this.neurons[n].setOutput(outputs[n])
		}
	}
	this.getOutputs = function(){
		var outputs = []
		for(var n = 0; n < this.neurons.length; n++){
			outputs.push(this.neurons[n].output)
		}
		return outputs
	}
	this.forward = function(){
		for(var n = 0; n< this.neurons.length; n++){
			this.neurons[n].forward()
		}
	}
	this.update = function(learningRate){
		for(var n = 0; n<this.neurons.length; n++){
			this.neurons[n].update(learningRate)
		}
	}
	
}

function Neuron(prevLayer){
	this.prevLayer = prevLayer
	this.output = 0
	this.error = 0
	this.weights = []
	this.bias = new weight(Math.random())
	if(this.prevLayer != undefined){
		for(var n = 0; n< prevLayer.neurons.length; n++){
			this.weights.push(new weight(Math.random()))
		}
	}

	this.setOutput = function(output){

		this.output = output
	}

	this.forward = function(){
		if(this.prevLayer == undefined){
			return;
		}
		this.output = 0
		for(var p = 0; p< prevLayer.neurons.length; p++){
			var connectedNeuron = this.prevLayer.neurons[p]
			this.output += connectedNeuron.output * this.weights[p].weight
		}
		this.output += this.bias.weight //add bias element
		this.output = Sigmoid(this.output)

	}
	this.update = function(learningRate){
		this.bias.weight += this.bias.delta * learningRate
		this.bias.delta = 0
		for(var w = 0; w< this.weights.length; w++){
			this.weights[w].weight += this.weights[w].delta * learningRate
			//console.log(this.weights[w].delta )
			this.weights[w].delta = 0
		}
	}

}
function weight(weight) {
	this.weight = weight;
	this.delta =  0;
}
//activation function
function Sigmoid(x) {//note for sigmoid prime we use sig * (1-sig), continued...
	return 1 / (1 + Math.exp(-x));// in the case the function has already been run for sig, just use x*(1-x)
}

var layout = [2,3,1]
var net = new Network(layout)
for(var i = 0; i< 2000; i++){
	net.train([1,1],[0])
	net.train([1,0],[1])
	net.train([0,1],[1])
	net.train([0,0],[0])
	
}



console.log("res")
console.log(net.forward([1,1]))
console.log(net.forward([1,0]))
console.log(net.forward([0,1]))
console.log(net.forward([0,0]))





