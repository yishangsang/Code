import java.io.*;

/**
 * Created by PC on 2016/11/14.
 */
public class Produce_arff {
    public static void main(String[] args)
    {
        try {
            String in_path = "G:\\实验室\\半监督 代码数据\\semi_train\\train_1\\";
            String out_path = "G:\\实验室\\半监督 代码数据\\semi_train\\result\\";
            BufferedWriter label_w = new BufferedWriter(new FileWriter(out_path + "train\\labeled_6.arff"));
            BufferedWriter unlabel_w = new BufferedWriter(new FileWriter(out_path + "train\\unlabeled_6.arff"));

            //生成新的label文件
            BufferedReader Reader = new BufferedReader(new FileReader(in_path+"labeled_5.arff"));
            String lineTxt = "";
            while((lineTxt = Reader.readLine())!=null){
                label_w.write(lineTxt);
                label_w.newLine();
            }
            Reader.close();

            Reader = new BufferedReader(new FileReader(out_path+"highConfidence.txt"));
            lineTxt = "";
            while((lineTxt = Reader.readLine())!=null){
                if(lineTxt.substring(lineTxt.split(" %")[0].lastIndexOf(","),lineTxt.split(" %")[0].lastIndexOf(",")+1).equals("3")||lineTxt.substring(lineTxt.split(" %")[0].lastIndexOf(","),lineTxt.split(" %")[0].lastIndexOf(",")+1).equals("4"))
                    label_w.write(lineTxt.substring(0,lineTxt.split(" %")[0].lastIndexOf(","))+","+lineTxt.split("  ")[lineTxt.split("  ").length-1]+" %"+lineTxt.split(" %")[1]);
                else
                label_w.write(lineTxt);
                label_w.newLine();
            }
            Reader.close();

            Reader = new BufferedReader(new FileReader(out_path+"lowConfidence.txt"));
            lineTxt = "";
            while((lineTxt = Reader.readLine())!=null){
                label_w.write(lineTxt.substring(0,lineTxt.split(" %")[0].lastIndexOf(","))+","+lineTxt.split("  ")[lineTxt.split("  ").length-1]+" %"+lineTxt.split(" %")[1]);
                label_w.newLine();
            }
            label_w.close();
            Reader.close();


            //生成新的unlabel文件
            BufferedReader r = new BufferedReader(new FileReader(out_path+"middleConfidence.txt"));
            String str = "";
            unlabel_w.write("@relation ExceptionRelation\n\n"
                    + "@attribute pathLen numeric\n"
                    + "@attribute subPathNumber numeric\n"
                    + "@attribute subPathMaxLen numeric\n"
                    + "@attribute subPathAvgLen numeric\n"
                    + "@attribute pathType numeric\n"
                    + "@attribute paraLen numeric\n"
                    + "@attribute paraNum numeric\n"
                    + "@attribute paraAvgLen numeric\n"
                    + "@attribute paraNameType numeric\n"
                    + "@attribute paraNameMaxLen numeric\n"
                    + "@attribute paraValueType numeric\n"
                    + "@attribute paraValueMaxLen numeric\n"
                    + "@attribute digitPercent numeric\n"
                    + "@attribute alphaPercent numeric\n"
                    + "@attribute urlUnknownAmount numeric\n"
                    + "@attribute nginxTest numeric\n"
                    + "@attribute paraValueContainIp numeric\n"
                    + "@attribute sqlRiskLevel numeric\n"
                    + "@attribute xssRisklevel numeric\n"
                    + "@attribute sensitiveRisklevel numeric\n"
                    + "@attribute otherRisklevel numeric\n"
                    + "@attribute directoryMaxLength numeric\n"
                    + "@attribute classlabel {0,1,2,3,4,5}\n\n"
                    + "@data\n");
            while((str = r.readLine())!=null){
                unlabel_w.write(str);
                unlabel_w.newLine();
            }
            unlabel_w.close();
            r.close();

        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
}
