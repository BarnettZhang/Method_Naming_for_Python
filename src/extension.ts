import * as vscode from 'vscode';


export function activate(context: vscode.ExtensionContext) {
	// Generate method name command. Main function of the plug-in
	let disposable1 = vscode.commands.registerCommand('sourcery.generate', () => {
		
		var startTime = performance.now()
		const editor = vscode.window.activeTextEditor; 
		const selectedText = editor?.document.getText(editor?.selection)

		const spawn = require("child_process").spawn;
		const pythonProcess = spawn('python',[context.extensionPath + "/resources/predict.py", context.extensionPath + "/resources", selectedText]);
	

		pythonProcess.stdout.on('data', (data:any) => {
			vscode.window.showInformationMessage("The top 5 recommended method names are: " + data);
			var endTime = performance.now()
			var time = ((endTime - startTime) / 1000).toFixed(3);
			vscode.window.showInformationMessage("The time taken to execute the extension is: " + String(time) + "s");
		}); 
		
	});
	context.subscriptions.push(disposable1);

	// Test measure time. Test function to measure execution time for each steps in order to collect data
	let disposable2 = vscode.commands.registerCommand('sourcery.measureTime', () => {
		var startTime = performance.now();
		const editor = vscode.window.activeTextEditor; 
		const selectedText = editor?.document.getText(editor?.selection);

		const spawn = require("child_process").spawn;
		const pythonProcess = spawn('python',[context.extensionPath + "/resources/measureExecutionTime.py", context.extensionPath + "/resources", selectedText]);
	

		pythonProcess.stdout.on('data', (data:any) => {
			vscode.window.showInformationMessage(String(data));
			var endTime = performance.now();
			var time = ((endTime - startTime) / 1000).toFixed(3);
			vscode.window.showInformationMessage("The time taken to execute the extension is: " + String(time) + "s");
		}); 
		
	});

	context.subscriptions.push(disposable2);

	let disposable3 = vscode.commands.registerCommand('sourcery.loadModel', () => {
		
		var startTime = performance.now()
		const spawn = require("child_process").spawn;
		const pythonProcess = spawn('python',[context.extensionPath + "/resources/loadModel.py", context.extensionPath + "/resources"]);
	

		pythonProcess.stdout.on('data', (data:any) => {
			vscode.window.showInformationMessage(String(data));
			var endTime = performance.now();
			var time = ((endTime - startTime) / 1000).toFixed(3);
			vscode.window.showInformationMessage("The time taken to execute the extension is: " + String(time) + "s");
		}); 
		
	});
	context.subscriptions.push(disposable3);

	let disposable4 = vscode.commands.registerCommand('sourcery.generateFromClient', () => {
		
		var startTime = performance.now()
		const editor = vscode.window.activeTextEditor; 
		const selectedText = editor?.document.getText(editor?.selection)

		const spawn = require("child_process").spawn;
		const pythonProcess = spawn('python',[context.extensionPath + "/resources/predictClient.py", selectedText]);
	

		pythonProcess.stdout.on('data', (data:any) => {
			vscode.window.showInformationMessage("The top 5 recommended method names are: " + data);
			var endTime = performance.now();
			var time = ((endTime - startTime) / 1000).toFixed(3);
			vscode.window.showInformationMessage("The time taken to execute the extension is: " + String(time) + "s");
		}); 
		
	});
	context.subscriptions.push(disposable4);

	let disposable5 = vscode.commands.registerCommand('sourcery.unloadModel', () => {
		

		const spawn = require("child_process").spawn;
		const pythonProcess = spawn('python',[context.extensionPath + "/resources/predictClient.py", "ThisIsToStopRunningTheModel"]);
	

		pythonProcess.stdout.on('data', (data:any) => {
			vscode.window.showInformationMessage(String(data));
		}); 
		
	});
	context.subscriptions.push(disposable4);
}


// this method is called when your extension is deactivated
export function deactivate() {}
