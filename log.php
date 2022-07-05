<?php
session_start();
include('connection.php');

if (isset($_POST['login']))
{	
$username=$_POST['email']; 
$password=$_POST['password']; 

	$sel="SELECT * FROM reg WHERE email='$username' and password='$password'";
	$result = mysqli_query($con,$sel) or die(mysql_error());
	$rows = mysqli_num_rows($result);
	$row=mysqli_fetch_array($rows);
	
	if($rows==1)
	{	
		$_SESSION['user']=$row['id'];
		header("location:upload.php");
		
	}
	else{
		
		header("location:index.php");
		
	}
}
	
?>
 
 



