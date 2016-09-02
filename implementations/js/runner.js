/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2014, Erick Lavoie, Faiz Khan, Sujay Kathrotia, Vincent
 * Foley-Bourgon, Laurie Hendren
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
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

    if (layer_size === expected_layer_size) {
        var sum_of_hidden_weights = 0;
        for (var i = 1; i <= net.hidden_n; ++i) {
            for (var j = 1; j <= net.output_n; ++j) {
                sum_of_hidden_weights += net.hidden_weights[i * (net.output_n + 1) + j];
            }
        }
        if (!(expected_sum_of_hidden_weights - eps < sum_of_hidden_weights &&
            sum_of_hidden_weights < expected_sum_of_hidden_weights + eps)) {
            throw new Error("ERROR: expected a sum of hidden weights of '" + expected_sum_of_hidden_weights + "'" +
                " for an input size of '" + expected_layer_size + "'" +
                " but got '" + sum_of_hidden_weights + "' instead");
        }
    } else {
        console.log("WARNING: no self-checking for input size of '" + layer_size + "'");
    }

    //console.log("Output: " + net.output_units[1].toFixed(4) + "\t" + net.output_delta[1].toFixed(4));
    net = null;
    console.log("Computation time: " + (time1 - time0) / 1000 + " s\n");
    console.log(JSON.stringify({
        status: 1,
        options: "run (" + layer_size + ")",
        time: (time1 - time0) / 1000,
        output: Math.floor(sum_of_hidden_weights/eps)
    }));
}
