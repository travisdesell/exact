import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;

import java.text.ParseException;
import java.text.SimpleDateFormat;

public class ParseTurbines {

    //time format is like: 2013-01-07T01:20:00+01:00
    public static SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX");
    // there are datapoints whose values at the last 16 columns are empty from the original data, and it would cause EXAMM fail at reading those files
    // since we are not using parameters which has empty values, the last 16 columns are not written into the files
    public static int num_column = 122;
    public static class TurbineLine {

        Date date;
        String date_time;
        String[] values;
        boolean isEmpty = false;
        public TurbineLine(String[] values) throws ParseException {
            this.values = values;
            date = sdf.parse(values[1]);
            if (values[2].length() <= 0) {
                this.isEmpty = true;
            }
        }

        public long getDiffMinutes(Date lastDate) {
            long diff = date.getTime() - lastDate.getTime();
            long diffMinutes = diff / (60 * 1000) % 60;
            return diffMinutes;
        }

        public boolean writeTo(BufferedWriter bw, Date lastDate, String turbine) throws IOException {
            boolean weirdMinutes = false;
            // there are datapoints from the same time point
            // so check if the time difference is 0 brfore print this line, if so, skip writing this line to file
            if (lastDate != null) {
                long diffMinutes = getDiffMinutes(lastDate);       
                if (diffMinutes == 0) {
                    System.out.println("turbine '" + turbine + ", " + date + ", minutes since last: " + diffMinutes);
                    return true;
                }

            }
            for (int i = 0; i < num_column; i++) {
                if (i != 0) bw.write(",");
                if (i == 1) {
                    if (lastDate != null) {
                        long diffMinutes = getDiffMinutes(lastDate);  
                        if (diffMinutes != 10) {
                            System.out.println("turbine '" + turbine + ", " + date + ", minutes since last: " + diffMinutes);
                            weirdMinutes = true;
                        }
                        bw.write(diffMinutes + ",");
                    } else {
                        bw.write(",");
                    }
                }
                bw.write(values[i]);
            }   
            bw.write("\n");

            return weirdMinutes;
        }

    }

    public static void writeHeaders(BufferedWriter bw, String[] columnNames) throws IOException {
        for (int i = 0; i < num_column; i++) {
            if (i != 0) bw.write(",");

            if (i == 1) {
                bw.write("Min_since_last,");
            }
            bw.write(columnNames[i]);
        }
        bw.write("\n");
    }

    public static void main(String[] arguments) {
        String filename = arguments[0];
        String addition = arguments[1];
        String[] columnNames;

        try {
            //create a buffered reader given the filename (which requires creating a File and FileReader object beforehand)
            BufferedReader bufferedReader = new BufferedReader(new FileReader(new File(filename)));

            System.out.println("'" + filename + "' contents are:");

            //the first line is the column names
            String readLine = bufferedReader.readLine();
            columnNames = readLine.split(";", Integer.MAX_VALUE);
            System.out.println("Column names are: " + Arrays.toString(columnNames));

            ArrayList<String> valueLines = new ArrayList<String>();

            HashMap<String, ArrayList<TurbineLine>> map = new HashMap<String, ArrayList<TurbineLine>>();

            //read the file line by line and add them to the valueLines array
            //list so we know how large to make our value array
            while ((readLine = bufferedReader.readLine()) != null) {
                String[] values = readLine.split(";", Integer.MAX_VALUE);

                if (map.get(values[0]) == null) {
                    System.out.println("Adding new turbine: '" + values[0] + "'");
                    map.put(values[0], new ArrayList<TurbineLine>());
                }

                if (values.length != columnNames.length) {
                    System.err.println("values.length: " + values.length + " != columnNames.length: " + columnNames.length);
                    System.err.println(readLine);
                    System.err.println(Arrays.toString(values));
                    System.exit(1);
                }

                map.get(values[0]).add(new TurbineLine(values));
            }


            for (String key : map.keySet()) {
                ArrayList<TurbineLine> lines = map.get(key);
                System.out.println("Sorting turbine '" + key + "' by date and time");

                Collections.sort(lines, new Comparator<TurbineLine>() {
                    public int compare(TurbineLine o1, TurbineLine o2) {
                        return o1.date.compareTo(o2.date);
                    }
                });

                int fileNumber = 1;

                System.out.println("Turbine '" + key + "' had " + lines.size() + " lines.");
                BufferedWriter bw = new BufferedWriter(new FileWriter(new File("turbine_" + key + "_" + addition + "_" + fileNumber + ".csv")));

                writeHeaders(bw, columnNames);

                int maxLines = 55000;
                int lineCount = 0;
                Date lastDate = null;
                boolean file_closed = false;
                TurbineLine prevLine = null;
                for (TurbineLine line : lines) {
                    if (line.isEmpty || lineCount >= maxLines) {
                        if (!file_closed) {
                            bw.close();
                            file_closed = true;
                        }
                        continue;
                        // bw = new BufferedWriter(new FileWriter(new File("turbine_" + key + "_" + addition + "_" + fileNumber + ".csv")));
                        // writeHeaders(bw, columnNames);
                    }

                    if (file_closed) {
                        if (lineCount > 10){
                            fileNumber++;
                        }
                        lastDate = null;
                        prevLine = null;
                        lineCount = 0;
                        bw = new BufferedWriter(new FileWriter(new File("turbine_" + key + "_" + addition + "_" + fileNumber + ".csv")));
                        writeHeaders(bw, columnNames);
                        file_closed = false;
                    }

                    Date date = line.date;

                    if (line.values.length != columnNames.length) {
                        System.exit(1);
                    }

                    if (line.writeTo(bw, lastDate, key + " " + addition)) {
                        //there was not 10 minutes between the last reading and this one
                        System.out.println("happend on line: " + lineCount);

                        System.out.println("\tcurrent, previous");
                        for (int i= 0; i < line.values.length; i++) {

                            System.out.println("\t" + line.values[i] + "," + prevLine.values[i]);
                        }
                        System.out.println();
                    }

                    lastDate = date;
                    prevLine = line;
                    lineCount++;
                }
                bw.close();
            }

            
        } catch (ParseException e) {
            System.err.println("ERROR opening DataSet file: '" + filename + "'");
            e.printStackTrace();
            System.exit(1);

        } catch (IOException e) {
            System.err.println("ERROR opening DataSet file: '" + filename + "'");
            e.printStackTrace();
            System.exit(1);
        }

    }
}
