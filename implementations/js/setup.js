function bpnn_internal_create(n_in, n_hidden, n_out) {
    //var newnet = Object.create(BPNN);

    this.input_n = n_in;
    this.hidden_n = n_hidden;
    this.output_n = n_out;
    this.input_units = new Float64Array(n_in + 1);
    this.hidden_units = new Float64Array(n_hidden + 1);
    this.output_units = new Float64Array(n_out + 1);

    this.hidden_delta = new Float64Array(n_hidden + 1);
    this.output_delta = new Float64Array(n_out + 1);
    this.target = new Float64Array(n_out + 1);

    this.input_weights = new Float64Array((n_in + 1) * (n_hidden + 1)); // TA
    this.hidden_weights = new Float64Array((n_hidden + 1) * (1 + n_out)); // TA

    this.input_prev_weights = new Float64Array((n_in + 1) * (1 + n_hidden));
    this.hidden_prev_weights = new Float64Array((n_hidden + 1) * (1 + n_out)); // TA

    return this;
}

function bpnn_randomize_array(w, m, n) {
    var i = 0,
        l = (m + 1) * (n + 1);

    for (i = 0; i < l; i++) {
        w[i] = Math.random();
    }
}

function loadInput(w, m, n) {
    var i = 1,
        l = (m + 1) * (n + 1);

    for (i = 1; i < l; i++) {
        w[i] = Math.random();
    }
}

function bpnn_randomize_row(w, m) {
    for (var i = 0; i <= m; i++) {
        w[i] = 0.1;
    }
}

function bpnn_create(n_in, n_hidden, n_out) {
    var newnet;

    newnet = new bpnn_internal_create(n_in, n_hidden, n_out);

    bpnn_randomize_array(newnet.input_weights, n_in, n_hidden);
    bpnn_randomize_array(newnet.hidden_weights, n_hidden, n_out);
    bpnn_randomize_row(newnet.target, n_out);

    // Load input image with random values
    loadInput(newnet.input_units, n_in, 1);

    return newnet;
}
