import csv
from tensorboardX import SummaryWriter

# Paths to your CSV files and TensorBoard log directories
results_csv_path_u1 = 'result_100_U1.csv' 
results_csv_path_u2 = 'result_100_U2.csv' 
log_dir_u1 = 'tensorboard/U1'
log_dir_u2 = 'tensorboard/U2'

# Function to log results from a CSV file
def log_results(csv_reader, writer):
    for epoch, row in enumerate(csv_reader):
        
        # Extract and calculate losses for training and validating
        # epochs = float(row['epochs'])
        train_crossentropyloss = float(row['train_crossentropyloss'])
        train_dice_score = float(row['train_dice_score'])
        train_dice_loss = float(row['train_dice_loss'])
        val_dice_score = float(row['val_dice_score'])
        val_dice_loss = float(row['val_dice_loss'])

        # Extract learning rate, precision, and recall
        lr = float(row['lr'])

        # Log the desired metrics to TensorBoard using the same scalar names
        writer.add_scalar('Train/CrossEntropyLoss', train_crossentropyloss, epoch)
        writer.add_scalar('Train/Dice_Score', train_dice_score, epoch)
        writer.add_scalar('Train/Dice_Loss', train_dice_loss, epoch)
        writer.add_scalar('Train/Scheduled_Learning_Rate', lr, epoch)

        writer.add_scalar('Validate/Dice_Score', val_dice_score, epoch)
        writer.add_scalar('Validate/Dice_Loss', val_dice_loss, epoch)


# Log results from Y1
with open(results_csv_path_u1, mode='r') as file1:
    csv_reader1 = csv.DictReader(file1)
    writer_y1 = SummaryWriter(log_dir=log_dir_u1)
    log_results(csv_reader1, writer_y1)
    writer_y1.close()

# Log results from Y2
with open(results_csv_path_u2, mode='r') as file2:
    csv_reader2 = csv.DictReader(file2)
    writer_y2 = SummaryWriter(log_dir=log_dir_u2)
    log_results(csv_reader2, writer_y2)
    writer_y2.close()
