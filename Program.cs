namespace NeuralNetwork_Test
{
    public static class Program
    {
        public static void Main()
        {
            var net = new NeuralNetwork(new[] { 2, 3, 1 });

            // XOR dataset for testing (not MNIST)
            var trainingData = new List<(double[] x, double[] y)>
        {
            (new double[] { 0, 0 }, new double[] { 0 }),
            (new double[] { 0, 1 }, new double[] { 1 }),
            (new double[] { 1, 0 }, new double[] { 1 }),
            (new double[] { 1, 1 }, new double[] { 0 })
        };

            net.SGD(trainingData, epochs: 10000, miniBatchSize: 4, eta: 3.0);

            foreach (var (x, _) in trainingData)
            {
                var result = net.FeedForward(x)[0];
                Console.WriteLine($"{x[0]} XOR {x[1]} = {Math.Round(result)} ({result:F3})");
            }
        }
    }
}



