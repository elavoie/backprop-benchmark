if (typeof performance === "undefined") {
    performance = Date;
}

Math.random = Math.commonRandomJS;

function runner(layer_size) {
    var net;
    var out_err, hid_err;
    var time0, time1;
    var expected_layer_size = 2850000;
    var expected_sum_of_hidden_weights = 10.855641469359398;
    var eps = 0.00001;
    net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)

    //entering the training kernel, only one iteration
    time0 = performance.now();
    bpnn_train_kernel(net);
    time1 = performance.now();

    var sum_of_hidden_weights = 0;
    for (var i = 1; i <= net.hidden_n; ++i) {
        for (var j = 1; j <= net.output_n; ++j) {
            sum_of_hidden_weights += net.hidden_weights[i * (net.output_n + 1) + j];
        }
    }

    var ADJUST = 0.1;
    net = null;
    console.log("Computation time: " + (time1 - time0) / 1000 + " s\n");
    console.log(JSON.stringify({
        status: 1,
        options: "run (" + layer_size + ")",
        time: (time1 - time0) / 1000,
        output: Math.floor(sum_of_hidden_weights*layer_size*ADJUST)
    }));
}
