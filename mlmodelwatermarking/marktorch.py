import random
import warnings
from math import floor

import numpy as np
import random
import pyfiglet
import torch
import tqdm
from cryptography.fernet import Fernet
from torch.utils.data import DataLoader
import torch.optim as optim

from mlmodelwatermarking.loggers.logger import logger
from mlmodelwatermarking.verification import verify

warnings.filterwarnings('ignore')


class Trainer:
    def __init__(
                self,
                model,
                args,
                trainset,
                valset,
                testset,
                specialset=None):

        """ Main wrapper class to watermark Pytorch models.

        Args:
            model (Object): model to be trained and watermarked
            trainset (Object): training dataset
            valset (Object): validation dataset
            testset (Object): test dataset
            specialset (Object, optional): trigger set for 'selected'

            
            args (dict): args for watermarking

        """

        # Non optional
        self.model = model
        self.trainset = trainset
        self.args = args
        self.valset = valset
        self.testset = testset
        self.specialset = specialset

        self.watermark = args.watermark

        if args.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
        self.patch_args = args.trigger_patch_args

        if args.gpu:
            available = torch.cuda.is_available()
            self.device = torch.device('cuda' if available else 'cpu')
        else:
            self.device = 'cpu'
        if self.args.verbose is False:
            logger.disable()
        if self.watermark:
            logger.info('Generation of the trigers')
            loaders = self.generate_trigger()
            (self.trainloader, self.valloader,
                self.testloader, self.triggerloader) = loaders
        else:
            self.trainloader, self.valloader, self.testloader = self.loaders()

    def loaders(self, trainset=None):
        """Load the dataloaders for training.

        Args:
            trainset (Object, optional): trainset if watermark
                activated

        Returns:
            trainloader (Object): training loader
            valloader (Object): validation loader
            testloader (Object): test loader

        """
        if not trainset:
            trainset = []
            for x, y in self.trainset:
                trainset.append((x, y))

        trainloader = DataLoader(
                                trainset,
                                batch_size=self.args.batch_size,
                                shuffle=True)
        testloader = DataLoader(
                                self.testset,
                                batch_size=self.args.batch_size,
                                shuffle=True)
        valloader = DataLoader(
                                self.valset,
                                batch_size=self.args.batch_size,
                                shuffle=True)

        return trainloader, valloader, testloader

    def generate_trigger(self):
        """Generate trigger data for watermark.

        Returns:
            loaders (list): dataloaders with triggers
        """

        if self.args.trigger_technique == 'noise':
            loaders = self.generate_trigger_noise()
        elif self.args.trigger_technique == 'selected':
            loaders = self.generate_trigger_selected()
        elif self.args.trigger_technique == 'patch':
            loaders = self.generate_trigger_patch_msg()
        elif self.args.trigger_technique == 'merrer':
            loaders = self.generate_trigger_merrer()
        else:
            raise NotImplementedError

        return loaders

    def generate_trigger_noise(self):
        """Generation random trigger set.

        Returns:
            trainloader (Object): training loader
            valloader (Object): validation loader
            testloader (Object): test loader
            triggerloader (Object): trigger loader
        """
        batch_x, _ = self.trainset[0]
        shapes = list(batch_x.shape)

        # Compute the triggers
        WM_X = torch.randn([self.args.trigger_size] + shapes)
        labels = [int(i) for i in range(self.args.nbr_classes)]
        WM_y = random.choices(labels, k=self.args.trigger_size)

        watermarked_dataset, triggerset = [], []
        for x, y in self.trainset:
            watermarked_dataset.append((x, y))
        for x, y in zip(WM_X, WM_y):
            triggerset.append((x, y))
            watermarked_dataset.append((x, y))

        trainloader, valloader, testloader = self.loaders(watermarked_dataset)
        triggerloader = torch.utils.data.DataLoader(
            triggerset, batch_size=self.args.trigger_size, shuffle=True)

        return trainloader, valloader, testloader, triggerloader

    def generate_trigger_selected(self):
        """Generation trigger set with specific data.

        Returns:
            trainloader (Object): training loader
            valloader (Object): validation loader
            testloader (Object): test loader
            triggerloader (Object): trigger loader
        """
        watermarked_dataset = []
        for x, y in self.trainset:
            watermarked_dataset.append((x, y))
        for x, y in self.specialset:
            watermarked_dataset.append((x, y))

        trainloader, valloader, testloader = self.loaders(watermarked_dataset)
        triggerloader = torch.utils.data.DataLoader(
            self.specialset, batch_size=self.args.trigger_size, shuffle=True)

        return trainloader, valloader, testloader, triggerloader

    def __write_letter(self, letter, data, offset):
        """Write text on input data through pyfliglet library.

        Returns:
            data (Object): data with inserted msg
        """
        result = pyfiglet.figlet_format(letter, font="3x5")
        result = result.replace('#', '1')
        result = result.replace(' ', '0')
        height = result.count('\n')
        result = result.replace('\n', '0')
        width = 4
        x, y = offset

        for i in range(height):
            for j in range(width):
                if int(result[i * (height - 1) + j]) == 1:
                    data[i + x][j + y] = 1

        return data

    def generate_trigger_patch_msg(self):
        """Generation trigger set with msg written
           on the legitimate input

        Returns:
            trainloader (Object): training loader
            valloader (Object): validation loader
            testloader (Object): test loader
            triggerloader (Object): trigger loader
        """
        watermarked_dataset = []
        for x, y in self.trainset:
            watermarked_dataset.append((x, y))
        # Sample the legitimate data to backdoor
        sampled_triggerset = random.choices(watermarked_dataset, k=20)
        triggerset = []
        # Load parameters
        msg = self.patch_args['msg']
        target = self.patch_args['target']
        for item_x, _ in sampled_triggerset:
            k = 0
            batch, size_x, size_y = item_x.shape
            item_x = item_x.reshape(size_x, size_y)
            for c in msg:
                item_x = self.__write_letter(c, item_x, offset=(0, k))
                k += 4
            # Update trigger set and training data
            reshape_item = item_x.reshape(batch, size_x, size_y)
            triggerset.append((reshape_item, target))
            watermarked_dataset.append((reshape_item, target))

        # Update loaders
        trainloader, valloader, testloader = self.loaders(watermarked_dataset)
        triggerloader = torch.utils.data.DataLoader(
            triggerset, batch_size=self.args.batch_size, shuffle=True)

        return trainloader, valloader, testloader, triggerloader

    def __fgsm_attack(self, data, epsilon, data_grad):
        """ Compute adversarial example with the
            "fast gradient sign" method.

        Args:
            data (object): original data to be perturbated
            epsilon (float): epsilon parameter for perturbation
            data_grad (object): gradient for perturbation

        Returns:
            perturbed_data (Object): data with inserted msg
        """
        sign_data_grad = data_grad.sign()
        perturbed_data = data + epsilon * sign_data_grad
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        return perturbed_data

    def generate_trigger_merrer(self):
        """ Generation trigger set, based on

            Adversarial Frontier Stitching for
            Remote Neural Network Watermarking

            by Le Merrer et al. (2017)

        Returns:
            trainloader (Object): training loader
            valloader (Object): validation loader
            testloader (Object): test loader
            triggerloader (Object): trigger loader
        """

        epsilon = self.args.epsilon
        triggerset = []
        self.model.to(self.device)

        sampled = torch.utils.data.Subset(
            self.trainset, random.choices(range(len(self.trainset)), k=500))
        trainloader = torch.utils.data.DataLoader(
            sampled, batch_size=1, shuffle=True)

        for data, target in trainloader:

            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            # If the initial prediction is wrong, pass
            if init_pred.item() != target.item():
                continue

            # Calculate the loss
            loss = self.args.criterion(output, target)
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            perturbed_data = self.__fgsm_attack(data, epsilon, data_grad)
            output = self.model(perturbed_data)

            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() != target.item():
                shape = perturbed_data.shape
                triggerset.append((perturbed_data.reshape(shape[1:]),
                                   final_pred.item()))

        triggerloader = torch.utils.data.DataLoader(
            triggerset, batch_size=self.args.batch_size, shuffle=True)

        trainloader, valloader, testloader = self.loaders()
        return trainloader, valloader, testloader, triggerloader

    def train_step(self, X, Y, idx):
        """Training step

        Args:
            X (Tensor): input data
            Y (Tensor): label data
            idx (int): batch id

        """
        # Compute loss for original data
        inputs, labels = X.to(self.device), Y.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.args.criterion(outputs, labels)
        # Watermark loss if required
        if self.watermark:
            if idx % self.args.interval_wm == 0:
                wmloss = self.watermark_loss()
                loss += wmloss
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def watermark_loss(self):
        """Compute loss on watermark.

        Returns:
            wmloss (float): loss on watermark data

        """
        wmloss = None
        for _, data in enumerate(self.triggerloader):
            inputs, labels = data
            outputs = self.model(inputs.to(self.device))
            if not wmloss:
                wmloss = self.args.criterion(outputs, labels.to(self.device))
            else:
                wmloss += self.args.criterion(outputs, labels.to(self.device))
        return wmloss

    def get_ownership(self):
        """Compute ownership information

        Returns:
            ownership (dict): information on trigger data

        """
        ownership = {}
        WM_X, WM_y = [], []
        for _, data in enumerate(self.triggerloader):
            inputs, labels = data
            WM_X += list(inputs.cpu().detach().numpy())
            WM_y += list(labels.cpu().detach().numpy())

        ownership['inputs'] = WM_X
        ownership['labels'] = WM_y
        ownership['bounds'] = (min(WM_y), max(WM_y))

        return ownership

    def train(self):
        """Train the model on watermarked data.

        Args:
            trainloader (object): training data
            epochs (int): number of epochs

        Returns:
            ownership (dict): Ownership information (None
                if model not watermarked)

        """
        self.model.to(self.device)
        logger.info('Training')
        pbar = tqdm.tqdm(range(self.args.epochs),
                         disable=not self.args.verbose)
        for _ in pbar:
            for idx, data in enumerate(self.trainloader):
                X, y = data
                self.train_step(X, y, idx)

            acc_validation = round(100 * self.validate(), 4)
            description = f'Validation accuracy: {acc_validation}'
            pbar.set_description_str(description)

        if self.args.encryption:
            logger.info('Encryption of the triggers')
            return self.encrypt()

        if self.watermark:
            return self.get_ownership()
        else:
            return None

    def validate(self):
        """Validate the accuracy after each epoch.

        Returns:
            accuracy (float): accuracy on validation set

        """
        correct = 0
        with torch.no_grad():
            self.model.eval().to(self.device)
            for x, y in self.valloader:
                probs = self.model(x.to(self.device))
                predictions = torch.argmax(probs, 1).cpu().numpy()
                correct += len(np.where(predictions == y.numpy())[0])
        return correct / len(self.valset)

    def test(self):
        """Test the accuracy of the data.

        Returns:
            accuracy (float): accuracy on test set

        """
        correct = 0
        with torch.no_grad():
            self.model.eval().to(self.device)
            for x, y in self.testloader:
                probs = self.model(x.to(self.device))
                predictions = torch.argmax(probs, 1).cpu().numpy()
                correct += len(np.where(predictions == y.numpy())[0])
        return correct / len(self.testset)

    def verify(self, ownership, suspect=None):
        """Verify if a given model is stolen.

        Args:
            ownership (dict): ownership information
            suspect (Object): the suspect model (None
                if model not watermarked)

        Returns:
            is_stolen (bool): is the model stolen ?

        """

        trigger_input = DataLoader(
                                    np.array(ownership['inputs']),
                                    batch_size=self.args.trigger_size,
                                    shuffle=False)
        pred_suspect = []
        predictions_reference = []

        # Verification with a suspect model
        if suspect:
            logger.info('Comparing with suspect model')
            with torch.no_grad():
                suspect.eval().to(self.device)
                self.model.eval().to(self.device)
                for _, batch in enumerate(trigger_input):
                    p_suspect = torch.argmax(suspect(batch.to(self.device)), 1)
                    pred_suspect += list(p_suspect.cpu().numpy())
                    p_ref = torch.argmax(self.model(batch.to(self.device)), 1)
                    predictions_reference = list(p_ref.cpu().numpy())

        # Self-verification
        else:
            logger.info('Self-verification')
            suspect = self.model.to(self.device)
            with torch.no_grad():
                suspect.eval()
                for _, batch in enumerate(trigger_input):
                    p_suspect = torch.argmax(suspect(batch.to(self.device)), 1)
                    pred_suspect += list(p_suspect.cpu().numpy())
            predictions_reference = ownership['labels']

        verification = verify(
                            pred_suspect,
                            predictions_reference,
                            bounds=None,
                            number_labels=self.args.nbr_classes)
        return verification

    def encrypt(self, nb_blocks=5):
        """Store the watermark in encrypted fashion.

        Args:
            nb_blocks (int): Split of encrypted triggers

        Returns:
            encrypted_trigger (dict): Encrypted triggers information

        """
        # Initiating the keys and the triggers(crypted)
        keys = []
        triggers = {}

        ownership = self.get_ownership()
        WM_X = ownership['inputs']
        shape_x, shape_y = WM_X[0].shape
        triggers['dtype'] = WM_X.dtype
        nb_blocks = min(nb_blocks, len(WM_X))

        step = floor(len(WM_X) / nb_blocks)
        # For each block
        for block in range(0, nb_blocks):
            # Generate the encryption key
            key = Fernet.generate_key()
            # Add the key to the key set
            keys.append(key)
            # Initiating the encryption object based on the key
            f = Fernet(key)
            # Encrypt / store
            to_encrypt = WM_X[block * step:(block + 1) * step].tobytes()
            encrypted_trigger = f.encrypt(to_encrypt)
            triggers['block_' + str(block)] = encrypted_trigger

        triggers['shape'] = (int(shape_x / nb_blocks), shape_y)
        triggers['dtype'] = WM_X.dtype

        encrypted_trigger = {'triggers': triggers, 'keys': keys}

        return encrypted_trigger

    def decrypt_trigger(self, triggers, block_id, key):
        """ Decrypt trigger block

        Args:
            triggers (dict): Encrypted trigger information
            block_id (int): Trigger block to decrypt
            key (str): Decryption key

        Returns:
            clear_trigger (array): Decrypted trigger

        """
        shape = triggers['shape']
        dtype = triggers['dtype']
        f = Fernet(key)
        clear_trigger = f.decrypt(triggers['block_' + str(block_id)])
        clear_trigger = np.frombuffer(clear_trigger, dtype=dtype)
        return clear_trigger.reshape(shape)
