import * as vscode from 'vscode';


export function activate(context: vscode.ExtensionContext) {

	let disposable = vscode.commands.registerCommand('sourcery.generate', () => {
		var startTime = performance.now()
		const editor = vscode.window.activeTextEditor; 
		const selectedText = editor?.document.getText(editor?.selection)

		const spawn = require("child_process").spawn;
		const pythonProcess = spawn('python',[context.extensionPath + "/src/predict.py", context.extensionPath + "/src", selectedText]);
	

		pythonProcess.stdout.on('data', (data:any) => {
			vscode.window.showInformationMessage("The top 5 recommended method names are: " + data);
			var endTime = performance.now()
			var time = ((endTime - startTime) / 1000).toFixed(3);
			vscode.window.showInformationMessage("The time taken to execute the extension is: " + String(time) + "s");
		}); 
		
	});

	context.subscriptions.push(disposable);
}

// this method is called when your extension is deactivated
export function deactivate() {}
