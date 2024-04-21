import java.io.IOException;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.chain.ChainMapper;

public class WordCountChain {

  public static class LowerCaseMapper 
      extends Mapper<Object, Text, IntWritable, Text> {

    private Text lowercased = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        lowercased.set(value.toString().toLowerCase());
        context.write(new IntWritable(1), lowercased);
    }
  }

  public static class SpecialSymbolRemoverMapper
      extends Mapper<IntWritable, Text, Text, IntWritable> {

    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();
    private final static Pattern specialCharsPattern = Pattern.compile("[^a-zA-Z]");

    public void map(IntWritable key, Text value, Context context
                    ) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        String token = itr.nextToken().toLowerCase(); // Convert to lowercase for consistency
        token = specialCharsPattern.matcher(token).replaceAll(""); // Remove special characters
        if (!token.isEmpty()) { // Ignore empty tokens
          word.set(token);
          context.write(word, one);
        }
      }
    }
  }

  public static class TokenizerMapper
       extends Mapper<IntWritable, Text, Text, IntWritable>{

    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    public void map(IntWritable key, Text value, Context context
                    ) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        word.set(itr.nextToken()); 
        context.write(word, one);
      }
    }
  }

  public static class IntSumReducer
       extends Reducer<Text,IntWritable,Text,IntWritable> {
    private TreeMap<Text, IntWritable> topN = new TreeMap<>(); // TreeMap to store top N words
    private final int N = 10; // Top N words

    public void reduce(Text key, Iterable<IntWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      // Add to the TreeMap
      topN.put(new Text(key), new IntWritable(sum));
      // Keep only top N elements
      if (topN.size() > N) {
        topN.remove(topN.firstKey());
      }
    }

    @Override
    protected void cleanup(Context context) throws IOException,
            InterruptedException {
      // Output the top N words
      for (Text i : topN.keySet()) {
        context.write(i, topN.get(i));
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "word count");
    job.setJarByClass(WordCountChain.class);

    // Configure the chain of mappers
    Configuration lowerCaseMapperConf = new Configuration(false);
    ChainMapper.addMapper(job,
      LowerCaseMapper.class,
      Object.class, Text.class,
      IntWritable.class, Text.class,
      lowerCaseMapperConf);

    Configuration specialSymbolRemoverConf = new Configuration(false);
    ChainMapper.addMapper(job,
      SpecialSymbolRemoverMapper.class,
      IntWritable.class, Text.class,
      Text.class, IntWritable.class,
      specialSymbolRemoverConf);

    Configuration tokenizerConf = new Configuration(false);
    ChainMapper.addMapper(job,
      TokenizerMapper.class,
      IntWritable.class, Text.class,
      Text.class, IntWritable.class,
      tokenizerConf);

    // Set reducer and output classes
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);

    // Input and output paths
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));

    // Exit status
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
