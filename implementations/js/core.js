/*** The squashing function.  Currently, it's a sigmoid. ***/
function squash(x) {
    return (1.0 / (1.0 + Math.exp(-x)));
}

function bpnn_layerforward(l1, l2, conn, n1, n2) {
    var sum;
    var j, k;

    var nc = n2 + 1,
        nr = n1 + 1;

    /*** Set up thresholding unit ***/
    l1[0] = 1.0;
    /*** For each unit in second layer ***/
    for (j = 1; j < nc; j++) {
        /*** Compute weighted sum of its inputs ***/
        sum = 0.0;
        for (k = 0; k < nr; k++) {
            sum += conn[k * nc + j] * l1[k];
        }
        l2[j] = squash(sum);
    }
}

//extern "C"
function bpnn_output_error(delta, target, output, nj) {
    var o, t, errsum;
    errsum = 0.0;
    for (var j = 1; j <= nj; j++) {
        o = output[j];
        t = target[j];
        delta[j] = o * (1.0 - o) * (t - o);
        errsum += Math.abs(delta[j]);
    }
    return errsum;
}

function bpnn_hidden_error(delta_h, nh, delta_o, no, who, hidden) {
    var j, k;
    var h, sum, errsum;

    var nr = nh + 1,
        nc = no + 1;

    errsum = 0.0;
    for (j = 1; j < nr; j++) {
        h = hidden[j];
        sum = 0.0;
        for (k = 1; k < nc; k++) {
            sum += delta_o[k] * who[j * no + k];
        }
        delta_h[j] = h * (1.0 - h) * sum;
        errsum += Math.abs(delta_h[j]);
    }
    return errsum;
}

//var layer_size = 0;
var ETA = 0.3 //eta value
var MOMENTUM = 0.3 //momentum value

function bpnn_adjust_weights(delta, ndelta, ly, nly, w, oldw) {
    var new_dw;
    var k, j;
    var nr = nly + 1,
        nc = ndelta + 1;
    ly[0] = 1.0;
    for (j = 1; j < nc; j++) {
        for (k = 0; k < nr; k++) {
            new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k * nc + j]));
            w[k * nc + j] += new_dw;
            oldw[k * nc + j] = new_dw;
        }
    }
}

function bpnn_train_kernel(net) {
    var inp, hid, out;
    var out_err, hid_err;

    inp = net.input_n;
    hid = net.hidden_n;
    out = net.output_n;

    bpnn_layerforward(net.input_units, net.hidden_units, net.input_weights, inp, hid);
    bpnn_layerforward(net.hidden_units, net.output_units, net.hidden_weights, hid, out);

    out_err = bpnn_output_error(net.output_delta, net.target, net.output_units, out);
    hid_err = bpnn_hidden_error(net.hidden_delta, hid, net.output_delta, out, net.hidden_weights, net.hidden_units);

    bpnn_adjust_weights(net.output_delta, out, net.hidden_units, hid, net.hidden_weights, net.hidden_prev_weights);
    bpnn_adjust_weights(net.hidden_delta, hid, net.input_units, inp, net.input_weights, net.input_prev_weights);
}


