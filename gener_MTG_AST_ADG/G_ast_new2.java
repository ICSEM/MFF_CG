package parser.example;

import static java.util.Comparator.comparing;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Map;
import java.util.HashMap;
import java.util.List;


import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

//import parser.example.Main.getPackageName;

import java.util.logging.Logger;





public class G_ast_new2 {
    private static int ID=0;
    private static int fileNum=0;
    private static PrintWriter printWriter;
    private static StringBuilder info;
    private static StringBuilder arrow;
    private static StringBuilder stringBuilder;
    private static String packageName;  
    private static String fileName;
    private static String className;
    public static String inputDir="E:/dataSrcTest/wangxiao";
//  public static String inputDir="E:/dataSrcTest/test111";
    public static String outputfile="E:/output/output_ast1/total_ast1.txt";
    
    static File output=new File(outputfile);

    
    
    public static void parser1(Node n,int parentID){
    	if(n.getChildrenNodes().isEmpty()){
//    		String test1=String.valueOf(n);
//			if(test1!=null&&test1.length()>0){
//				ID++;
//				test1= test1.replaceAll("(?m)^\\s*$"+System.lineSeparator(), "");//去掉多个空行
//				test1=test1.replaceAll("(\r\n|\r|\n|\n\r)", "");//去掉回车
//				test1=test1.replaceAll("\\s{2,}", " ");//将多个连续的空格换成一个空格
//			}
//			info.append(ID+"["+test1+"\n");
//			info.append(test1+"\n");
//			System.out.println(8888888);
//    	
    	}else
    	{
    		
    		///正常生成AST树	`
    		for(Node sn:n.getChildrenNodes()){
//    			System.out.println(sn+"\n");
    			if(sn instanceof Comment ){
//    				System.out.print("这是个注释！");
//    				System.out.print(sn);
    			}
    			else{
    				
    				String test=String.valueOf(sn);
    				if(test!=null&&test.length()>0){
    					ID++;
    					test= test.replaceAll("(?m)^\\s*$"+System.lineSeparator(), "");//去掉多个空行
    					test=test.replaceAll("(\r\n|\r|\n|\n\r)", "");//去掉回车
    					test=test.replaceAll("\\s{2,}", " ");//将多个连续的空格换成一个空格
//    		            test= test.replaceAll("\\s*", "");
    		            arrow.append(parentID+"->"+ID+"\n");
    		            String[] sClass=sn.getClass().toString().split("\\.");
//    		            info.append(ID+sClass[sClass.length-1]+"\n");
//    		            info.append(ID+"class:"+sClass[sClass.length-1]+" info:"+test+"\n");
    		            if(sn.getChildrenNodes().isEmpty()){
//    		            	info.append(ID+"["+"class:"+sClass[sClass.length-1]+" info:"+test+" line:("+sn.getBegin()+","+sn.getEnd()+")"+"\n");
    		           
    		            	info.append(ID+"[(?)"+test+"\n");
    		            }
    		            else{
    		            	if(sn instanceof MethodDeclaration){
    		            		info.append(ID+"["+test+"\n");
    		            	}
    		            	else{
    		            	info.append(ID+"["+sClass[sClass.length-1]+"\n");
    		            	}
    		            }
    		            ///如果当前节点是，函数声明另外提取信息数据字典的存储
    		    		if(sn instanceof ClassOrInterfaceDeclaration){   
    		    				className=((ClassOrInterfaceDeclaration) sn).getName();
    		    		}
    		    		  
    		    		if(sn instanceof MethodDeclaration){   
	    	      			 stringBuilder.append(fileName+",");
	    	                 stringBuilder.append(packageName+",");
	    	                 stringBuilder.append(className+",");
	    	                 stringBuilder.append(ID+",");
	    	                 stringBuilder.append(((MethodDeclaration) sn).getName()+",");
	    	                 String temp=((MethodDeclaration) sn).getType()+"";
	    	                 if(temp.contains("/")){                	
	    	                 	String t[]=temp.split("\\n");
	    	                 	stringBuilder.append(t[t.length-1]+",");
	    	                 }
	    	                 else{
	    	                 	stringBuilder.append(((MethodDeclaration) sn).getType()+",");
	    	                 }
	    	                           
	    	                 NodeList<com.github.javaparser.ast.body.Parameter> parametersSmall = ((MethodDeclaration) sn).getParameters();
	    	                 if(parametersSmall.isEmpty()){    
	    	                 }
	    	                 else{                   	
	    	                 	for (com.github.javaparser.ast.body.Parameter parameter : parametersSmall) {
	    	                 		String parameterString=String.valueOf(parameter.getType());
	    	                 		String parametersList[]=parameterString.split("\\.");
	    	                 		
	    	                 		if(parametersList.length>0)
	    	                 		{                    		
	    	                 			parameterString=parametersList[parametersList.length-1];                   
	    	                 		}
	    	                 		
	    	                 		stringBuilder.append(parameterString+",");
	    	                     }     	
	    	                 	
	    	                 }                   
	    	                 stringBuilder.append("\n");
	    	    			}
//    		    		System.out.println(sn+"55555\n");
    		            parser1(sn,ID);    		         
    				}
    				
//    				System.out.println(String.valueOf(snNode.getClass().getTypeName())+String.valueOf(snNode.getBegin())+String.valueOf(snNode.getEnd()));
    				
    			}
    		}
    	}
    }
    	
    
    
    public static void traverseFolder(String path) {
		
        File file = new File(path);
        if (file.exists()) {
            File[] files = file.listFiles();
//          Collections.sort(files);
            System.out.println(files.length);
            

            for(int j=0;j<files.length;j++){
            	String path2;
            	path2 = path+"/"+j+".java";
//                	System.out.println(path2);
            	File file2 = new File(path2);
//                	System.out.println(file2);
//                	System.exit(0);
                
                if (file2.isDirectory()) {                
                    traverseFolder(file2.getAbsolutePath());
                } else {
                	if(file2.getName().endsWith(".java"))
                	{
                		try {
//                    			System.out.println(file2.getName().split("\\\\")[0]);
                			
                			info=new StringBuilder();
                		    arrow=new StringBuilder();
                		    fileName=file2.getName();
                		    packageName="";
                		    System.out.println(file2.getName());
							CompilationUnit cu = JavaParser.parse(file2);
//								System.out.println(cu);
							new getPackageName().visit(cu, null);
							String test=String.valueOf(cu);
                    		if(test!=null&&test.length()>0){    		            
//	                                test= test.replaceAll("\\s*", "");
                    			test= test.replaceAll("(?m)^\\s*$"+System.lineSeparator(), "");//去掉多个空行
            					test=test.replaceAll("(\r\n|\r|\n|\n\r)", "");//去掉回车
            					test=test.replaceAll("\\s{2,}", " ");//将多个连续的空格换成一个空格
                                String[] sClass=cu.getClass().toString().split("\\.");
                                //info.append(ID+sClass[sClass.length-1]+"\n");
                                info.append("AST\n");
                                
//	                                info.append(ID+"["+"class:"+sClass[sClass.length-1]+" info:"+test+" line:("+cu.getBegin()+","+cu.getEnd()+")"+"\n");
                                info.append(ID+"["+sClass[sClass.length-1]+"\n");
                                parser1(cu,ID);
                                ID++;
                    		}
                    		fileNum++;	                  
                    		saveOutput();
						} catch (FileNotFoundException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
                    	
                	}
            	}

        }
                
        }else {
        	System.out.println("文件不存在!");
        }
    }
    
    
    
    public static void saveOutput(){

    	
		try {
			
		if(!output.exists()) {   
			output.createNewFile();			
		}
		
	}
		catch(IOException e) {
			e.printStackTrace();
		}
		try {
		//FileOutputStream in=new FileOutputStream(a);
			   FileWriter fw = new FileWriter(output, true); //在文件末尾追加形式写文件
		       fw.write(info.toString());//写入字符串“我爱你”
		       fw.write(arrow.toString());
		       fw.write("\n");
		       fw.flush();//刷新缓存
		       fw.close();//关闭输入流
		       }catch(IOException e) {
					e.printStackTrace();
				}
    }
    private static class getPackageName extends VoidVisitorAdapter<Object>  {
        @Override
        public void visit(PackageDeclaration n, Object arg) {
            packageName=n.getPackageName();
            super.visit(n, arg);
        }
        
    }

    
    public static void main(String[] args) throws IOException {   
    	stringBuilder = new StringBuilder();
        String outputName="E:/output/output_dic/SignalResultsTest.csv";
        printWriter = new PrintWriter(new File(outputName));
        stringBuilder.append("File,Package,Class,ID,Method,ReturnType,Parameters\n");
        
    	if(output.exists()) {  
			output.delete();			
		}
    	traverseFolder(inputDir);	
    	printWriter.write(stringBuilder.toString());
    	printWriter.close();
        System.out.print("please check the file: "+outputName);
    	System.out.println("total file Num :"+fileNum);
    	System.out.print("The file has been parsed, please look at the path :"+outputfile);
    }
    

}

















