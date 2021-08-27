package com.yourorganization.maven_sample;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.Logger;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;

import javassist.bytecode.MethodInfo;

//搬运引用
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.stmt.SwitchEntry;
import com.github.javaparser.resolution.declarations.ResolvedMethodDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedTypeDeclaration;
import com.github.javaparser.resolution.types.ResolvedReferenceType;
import com.github.javaparser.resolution.types.ResolvedType;
import com.github.javaparser.symbolsolver.javaparsermodel.JavaParserFacade;
import com.github.javaparser.symbolsolver.model.resolution.SymbolReference;
import com.github.javaparser.symbolsolver.model.resolution.TypeSolver;

import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;

import static com.github.javaparser.StaticJavaParser.parse;
import static com.github.javaparser.symbolsolver.javaparser.Navigator.demandParentNode;
import static java.util.Comparator.comparing;



/**
 * Some code that uses JavaSymbolSolver.
 */
public class My_G_ADG1 {
//	public static Logger logger=Logger.getLogger("G_ADG111.class");
	private static PrintStream out = System.out;

	private static Map<Integer,Integer> edgeInfo;//建立新旧ID的关联
	private static Map<Integer,Integer> nodeInfo;
//	private static HashSet nodeInfo;
	private static String methodInfoFile="E:/output/output_dic/SignalResultsTest.csv";//上一个文件产生的函数签名信息
	private static String dataSrc="E:/dataSrcTest/test888";//数据集的位置
	private static String jarSrc="E:/导师任务/ADG任务代码/src_temp";//所需要jar包的位置
	private static String reasultPath="E:/output/output_adg/";//输出结果的路径
	private static String methodCallFile="E:/output/output_adg/MethodCall.csv";//存放提取到的函数调用信息的路径
		
	
	
	public static void createADGFile(String filePath,String fileName) throws IOException {	
		File dir = new File(filePath);
	    if (!dir.exists()) {
	        dir.mkdirs();
	    }
	    File checkFile = new File(filePath + fileName);
	    FileWriter writer = null;
	    try {
	     
	        if (!checkFile.exists()) {
	            checkFile.createNewFile();
	        }
	       
	        writer = new FileWriter(checkFile, false);
	        //写入文件头
	        writer.append("ADG\n");
	        
	        //写入结点信息
	    
	        for(Map.Entry<Integer,Integer>entry:nodeInfo.entrySet()){
	        	 writer.append(entry.getKey()+":"+entry.getValue()+"["+"\n");	        	
	        }
	        
	        
	        //写入边的关系信息
	        for(Map.Entry<Integer,Integer>entry:edgeInfo.entrySet()){
	        	writer.append(entry.getKey()+"->"+entry.getValue()+"\n");
	        }
	        	        	 	        
	        
	        writer.flush();
	    } catch (IOException e) {
	        e.printStackTrace();
	    } finally {
	        if (null != writer)
	            writer.close();
	    }
	}

	

	

	public static void readFile(){
		
		File csv=new File(methodInfoFile);
		BufferedReader br=null;
		try{
			br=new BufferedReader(new FileReader(csv));
		}catch(FileNotFoundException e){
			e.printStackTrace();
		}

		String line="";
		int lineNum=0;
		try{
			line=br.readLine();//忽略第一行
			int max_length=0;
			while((line=br.readLine())!=null){
				lineNum++;
				String readMethod[]=line.split(",");
				if(max_length<readMethod.length){
					max_length=readMethod.length;
				}	
				}
			System.out.println(lineNum);
			br=new BufferedReader(new FileReader(csv));
			String[][] arr = new String[lineNum][max_length];
			
			ArrayList<Integer> nexID=new ArrayList<>();
			
			line=br.readLine();//忽略第一行

			int line_num = 0;
			while((line=br.readLine())!=null){
				
				String readMethod[]=line.split(",");
				System.out.println(readMethod.length);
//				arr[line_num] = new String[readMethod.length];
				for(int i=0;i<readMethod.length;i++){		
//					arr[line_num][i]=new String(readMethod[i]);
					arr[line_num][i]=readMethod[i];
					System.out.println(arr[line_num][i]);
				}
				line_num=line_num+1;
//				System.out.println(arr[0][5]);
			}
			
//			System.out.println(arr[0][5]);
//			System.exit(0);
			
			int line_num_self=0;
			//将ReturnType与csv文件中的其他行的Parameters和class进行比较，看看之间是否存在依赖关系
			br=new BufferedReader(new FileReader(csv));
			line=br.readLine();//忽略第一行
			while((line=br.readLine())!=null){
				String readMethod[]=line.split(",");
				for(int j=0;j<lineNum;j++){
					//要求比较的两行，是不同行且属于相同的文件
					if(j!=line_num_self && readMethod[0].equals(arr[j][0])){
//						System.out.println(j);
						//如果ReturnType与其他行的class保持一致，ReturnType-->class
						if(readMethod[5].equals(arr[j][2])){
							//向txt文件中写入边的关系
							System.out.println("输出彼此有边的ID为："+readMethod[3]+' '+arr[j][3]);
//							perID.add(Integer.parseInt(readMethod[3]));
							String[] s1,s2;
							s1 = readMethod[0].split("\\.");
							s2 = arr[j][0].split("\\.");
							nodeInfo.put(Integer.parseInt(readMethod[3]),Integer.parseInt(s1[0]));
							nodeInfo.put(Integer.parseInt(arr[j][3]),Integer.parseInt(s2[0]));
							edgeInfo.put(Integer.parseInt(readMethod[3]), Integer.parseInt(arr[j][3]));
						}
						
						//如果比较的行有参数，则将ReturnType与Parameters比较，Parameters-->ReturnType
						else if(arr[j].length>6){
							int para_num= arr[j].length-6;
							for(int t=0;t<para_num;t++){
								int t1=6+t;
								if(readMethod[5].equals(arr[j][t1])&&!(arr[j][t1].equals("String")
										||arr[j][t1].equals("int")
										||arr[j][t1].equals("byte")
										||arr[j][t1].equals("short")
										||arr[j][t1].equals("long")
										||arr[j][t1].equals("float")
										||arr[j][t1].equals("double")
										||arr[j][t1].equals("boolean")
										||arr[j][t1].equals("char")
										)){
									//向txt文件中写入边的关系
									System.out.println("此处有边！");
									String[] s1,s2;
									s1 = readMethod[0].split("\\.");
									s2 = arr[j][0].split("\\.");
									nodeInfo.put(Integer.parseInt(readMethod[3]),Integer.parseInt(s1[0]));
									nodeInfo.put(Integer.parseInt(arr[j][3]),Integer.parseInt(s2[0]));
									edgeInfo.put(Integer.parseInt(arr[j][3]), Integer.parseInt(readMethod[3]));
									break;
								}
							}
							
						}
					}
				}
//				System.exit(0);
				line_num_self=line_num_self+1;
			}
//			System.exit(0);
			
		}catch(IOException e){
			
		}
	}



    public static void main(String[] args) throws IOException {
    
      String filePath=reasultPath;
      edgeInfo=new HashMap<Integer,Integer>();
	  nodeInfo=new HashMap<Integer,Integer>();
//      nodeInfo = new HashSet();
      
      readFile();

      createADGFile(filePath,"ADG.txt");

    
    }
}







