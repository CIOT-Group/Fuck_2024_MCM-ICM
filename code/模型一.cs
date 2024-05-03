using MathNet.Numerics;
public class Question1
{
    public (double, double) GetAverageValueAndStandardDeviation()
    {
        var result = (0.0, 0.0);
        List<int> list = new List<int> { };
        while (true)
        {
            var input = Console.ReadLine();
            if (input == "quit")
            {
                break;
            }

            list.Add(int.Parse(input));
            Console.WriteLine("您添加了一个数" + input);
        }

        result.Item1 = list.Average();
        result.Item2 = Math.Sqrt(list.Select(x => Math.Pow(x - result.Item1, 2)).Sum() / list.Count);
        return result;
    }

    public double GetCumulativeDistributionFunctionOfNormalDistribution(double z, double a, double s)
    {
        return 0.5 * (1 + SpecialFunctions.Erf((z - a) / (s * Math.Sqrt(2))));
    }

    public void Anticipate(double a)
    {
        List<double> res = new();
        double s = 0.4472136;
        Func<double, double> function = x =>
            SpecialFunctions.Erf(-Math.Pow((x - a), 2) / (2 * Math.Pow(s, 2))) / (Math.Sqrt(2 * Math.PI) * s);
        for (int i = 0; i < 5; i++)
        {
            res.Add(Integrate.GaussLegendre(function, i * 0.2f, (i + 1) * 0.2f));
        }

        List<double> res2 = new List<double>();
        foreach (var v in res)
        {
            res2.Add(Math.Round((v / res.Sum()), 4));
        }
        foreach (var result in res2)
        {
            Console.Write(result + " ");
        }
    }
}

public class Program
{
    public static void Main()
    {
        var question1 = new Question1();
        double a, s;
        Console.WriteLine("请输入数据计算均值和方差");
        (a, s) = question1.GetAverageValueAndStandardDeviation();
        Console.WriteLine("请输入 z");
        var z = double.Parse(Console.ReadLine());
        Console.WriteLine("您输入了值" + z);
        question1.Anticipate(question1.GetCumulativeDistributionFunctionOfNormalDistribution(z, a, s));
    }
}